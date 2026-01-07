"""
Swan AI Clone - Backend API
FastAPI server for lead tracking and processing
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import json
import sqlite3
import os
from datetime import datetime
import asyncio

app = FastAPI(title="Swan AI Clone API", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = "database/swan.db"

# Global settings (loaded from DB)
settings = {
    "ipinfo_api_key": "",
    "apollo_api_key": "",
    "hunter_api_key": "",
    "openai_api_key": "",
    "slack_webhook_url": "",
    "icp_config": {
        "industries": ["SaaS", "Technology", "E-commerce"],
        "min_employees": 50,
        "max_employees": 1000,
        "countries": ["United States", "United Kingdom", "Canada"],
        "target_titles": ["CEO", "CTO", "VP", "Director", "Head of"]
    }
}

# ============== MODELS ==============

class VisitorData(BaseModel):
    project_id: str
    session_id: str
    ip_address: Optional[str] = None
    current_url: str
    referrer: Optional[str] = ""
    pages_viewed: List[Dict[str, Any]] = []
    visit_duration: int = 0
    user_agent: Optional[str] = ""
    screen_size: Optional[str] = ""
    timestamp: str

class APISettings(BaseModel):
    ipinfo_api_key: Optional[str] = ""
    apollo_api_key: Optional[str] = ""
    hunter_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    slack_webhook_url: Optional[str] = ""

class ICPConfig(BaseModel):
    industries: List[str] = []
    min_employees: int = 50
    max_employees: int = 1000
    countries: List[str] = []
    target_titles: List[str] = []

class TestAPIRequest(BaseModel):
    api_type: str  # apollo, hunter, openai
    api_key: str
    test_domain: str = "notion.so"

# ============== DATABASE ==============

def init_db():
    """Initialize SQLite database"""
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Companies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT UNIQUE,
            name TEXT,
            industry TEXT,
            employee_count INTEGER DEFAULT 0,
            country TEXT,
            city TEXT,
            description TEXT,
            funding_stage TEXT,
            total_funding INTEGER DEFAULT 0,
            annual_revenue INTEGER DEFAULT 0,
            website TEXT,
            linkedin_url TEXT,
            enriched_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Contacts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            name TEXT,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            title TEXT,
            seniority TEXT,
            department TEXT,
            linkedin_url TEXT,
            phone TEXT,
            confidence INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies(id)
        )
    """)
    
    # Leads table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT UNIQUE,
            company_id INTEGER,
            session_id TEXT,
            ip_address TEXT,
            pages_viewed TEXT,
            visit_duration INTEGER DEFAULT 0,
            referrer TEXT,
            user_agent TEXT,
            icp_score INTEGER DEFAULT 0,
            intent_score INTEGER DEFAULT 0,
            tier TEXT DEFAULT 'cold',
            qualified BOOLEAN DEFAULT FALSE,
            match_reasons TEXT,
            miss_reasons TEXT,
            intent_signals TEXT,
            research_summary TEXT,
            talking_points TEXT,
            recommended_action TEXT,
            urgency TEXT DEFAULT 'low',
            email_draft TEXT,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (company_id) REFERENCES companies(id)
        )
    """)
    
    # Visitors table - ALL visitors with IP info
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            ip_address TEXT,
            city TEXT,
            region TEXT,
            country TEXT,
            country_code TEXT,
            org TEXT,
            hostname TEXT,
            is_business BOOLEAN DEFAULT FALSE,
            current_url TEXT,
            referrer TEXT,
            pages_viewed TEXT,
            visit_duration INTEGER DEFAULT 0,
            user_agent TEXT,
            screen_size TEXT,
            enriched BOOLEAN DEFAULT FALSE,
            lead_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def load_settings():
    """Load settings from database"""
    global settings
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        for key, value in rows:
            if key == "icp_config":
                settings[key] = json.loads(value)
            else:
                settings[key] = value
        conn.close()
    except:
        pass

def save_setting(key: str, value: Any):
    """Save a setting to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if isinstance(value, dict):
        value = json.dumps(value)
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()

# ============== API INTEGRATIONS ==============

async def get_ip_info(ip_address: str) -> Dict:
    """Get IP geolocation info using IPInfo API"""
    if not ip_address or ip_address in ["127.0.0.1", "localhost", ""]:
        return {
            "ip": ip_address,
            "city": "Local",
            "region": "Local",
            "country": "Local",
            "country_code": "XX",
            "org": "localhost",
            "hostname": "",
            "is_business": False
        }
    
    api_key = settings.get("ipinfo_api_key", "")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            url = f"https://ipinfo.io/{ip_address}/json"
            if api_key:
                url += f"?token={api_key}"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                org = data.get("org", "")
                
                # Check if business IP (not residential ISP)
                residential_keywords = ["Comcast", "Verizon", "AT&T", "T-Mobile", "Spectrum", 
                                       "Cox", "Charter", "Xfinity", "Virgin", "BT ", "Sky ",
                                       "Vodafone", "Airtel", "Jio", "BSNL", "Mobile", "Wireless",
                                       "Cellular", "Broadband"]
                is_business = bool(org) and not any(kw.lower() in org.lower() for kw in residential_keywords)
                
                return {
                    "ip": ip_address,
                    "city": data.get("city", "Unknown"),
                    "region": data.get("region", "Unknown"),
                    "country": data.get("country", "Unknown"),
                    "country_code": data.get("country", "XX"),
                    "org": org or "Unknown",
                    "hostname": data.get("hostname", ""),
                    "is_business": is_business,
                    "loc": data.get("loc", ""),
                    "timezone": data.get("timezone", "")
                }
        except Exception as e:
            print(f"IPInfo error: {e}")
    
    return {
        "ip": ip_address,
        "city": "Unknown",
        "region": "Unknown", 
        "country": "Unknown",
        "country_code": "XX",
        "org": "Unknown",
        "hostname": "",
        "is_business": False
    }

async def enrich_company_apollo(domain: str, api_key: str) -> Dict:
    """Enrich company data using Apollo API"""
    if not api_key:
        return {"error": "Apollo API key not configured"}
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                "https://api.apollo.io/v1/organizations/enrich",
                headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": api_key
                },
                json={"domain": domain}
            )
            
            if response.status_code == 200:
                data = response.json()
                org = data.get("organization", {})
                return {
                    "success": True,
                    "data": {
                        "name": org.get("name", ""),
                        "domain": org.get("primary_domain", domain),
                        "industry": org.get("industry", "Unknown"),
                        "employee_count": org.get("estimated_num_employees", 0),
                        "country": org.get("country", ""),
                        "city": org.get("city", ""),
                        "description": org.get("short_description", ""),
                        "funding_stage": org.get("latest_funding_stage", ""),
                        "total_funding": org.get("total_funding", 0),
                        "annual_revenue": org.get("annual_revenue", 0),
                        "website": org.get("website_url", ""),
                        "linkedin_url": org.get("linkedin_url", ""),
                        "technologies": org.get("technologies", []),
                        "raw": org
                    }
                }
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

async def find_contacts_hunter(domain: str, api_key: str) -> Dict:
    """Find contacts using Hunter.io API"""
    if not api_key:
        return {"error": "Hunter API key not configured", "contacts": []}
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"https://api.hunter.io/v2/domain-search",
                params={
                    "domain": domain,
                    "limit": 5,
                    "api_key": api_key
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                emails = data.get("data", {}).get("emails", [])
                contacts = []
                for e in emails[:5]:
                    contacts.append({
                        "name": f"{e.get('first_name', '')} {e.get('last_name', '')}".strip(),
                        "first_name": e.get("first_name", ""),
                        "last_name": e.get("last_name", ""),
                        "email": e.get("value", ""),
                        "title": e.get("position", ""),
                        "department": e.get("department", ""),
                        "seniority": e.get("seniority", ""),
                        "linkedin_url": e.get("linkedin", ""),
                        "confidence": e.get("confidence", 0),
                        "phone": e.get("phone_number", "")
                    })
                return {"success": True, "contacts": contacts}
            else:
                return {"success": False, "error": response.text, "contacts": []}
        except Exception as e:
            return {"success": False, "error": str(e), "contacts": []}

async def score_lead_openai(company: Dict, contacts: List, visit_data: Dict, api_key: str, icp_config: Dict) -> Dict:
    """Score lead using OpenAI"""
    if not api_key:
        return {"error": "OpenAI API key not configured"}
    
    prompt = f"""You are a B2B lead qualification AI. Analyze this website visitor and score them.

=== ICP CRITERIA ===
Industries: {', '.join(icp_config.get('industries', []))}
Company Size: {icp_config.get('min_employees', 50)} - {icp_config.get('max_employees', 1000)} employees
Countries: {', '.join(icp_config.get('countries', []))}
Target Titles: {', '.join(icp_config.get('target_titles', []))}

=== COMPANY DATA ===
Name: {company.get('name', 'Unknown')}
Domain: {company.get('domain', '')}
Industry: {company.get('industry', 'Unknown')}
Employees: {company.get('employee_count', 0)}
Country: {company.get('country', 'Unknown')}
Funding: {company.get('funding_stage', 'Unknown')}
Total Funding: ${company.get('total_funding', 0)}
Revenue: ${company.get('annual_revenue', 0)}
Description: {company.get('description', '')[:300]}

=== VISITOR BEHAVIOR ===
Pages Viewed: {json.dumps(visit_data.get('pages_viewed', []))}
Visit Duration: {visit_data.get('visit_duration', 0)} seconds
Referrer: {visit_data.get('referrer', 'Direct')}

=== CONTACTS FOUND ===
{chr(10).join([f"- {c.get('name', '')} | {c.get('title', '')} | {c.get('email', '')} | Confidence: {c.get('confidence', 0)}%" for c in contacts]) if contacts else 'No contacts found'}

=== SCORING RULES ===
- Score 0-100 based on ICP fit and intent signals
- HOT (80-100): Perfect ICP match + high intent
- WARM (50-79): Good fit, moderate intent
- COLD (0-49): Poor fit or low intent

=== OUTPUT JSON ONLY (no markdown) ===
{{
  "icp_score": <0-100>,
  "tier": "hot" | "warm" | "cold",
  "qualified": true | false,
  "match_reasons": ["reason1", "reason2"],
  "miss_reasons": ["reason1"],
  "intent_signals": ["signal1", "signal2"],
  "intent_score": <0-100>,
  "best_contact": {{
    "name": "...",
    "title": "...",
    "email": "...",
    "why": "reason"
  }},
  "recommended_action": "book_demo" | "send_email" | "add_nurture" | "skip",
  "urgency": "high" | "medium" | "low",
  "research_summary": "2-3 sentences for sales rep",
  "talking_points": ["point1", "point2"],
  "email_draft": {{
    "subject": "...",
    "body": "..."
  }}
}}"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a B2B lead qualification AI. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Clean and parse JSON
                content = content.replace("```json", "").replace("```", "").strip()
                result = json.loads(content)
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": response.text}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parse error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ============== LEAD PROCESSING ==============

def save_visitor_to_db(session_id: str, ip_info: Dict, visitor: VisitorData, enriched: bool = False, lead_id: str = None):
    """Save visitor to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO visitors 
        (session_id, ip_address, city, region, country, country_code, org, hostname, 
         is_business, current_url, referrer, pages_viewed, visit_duration, user_agent, 
         screen_size, enriched, lead_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        session_id,
        ip_info.get("ip", ""),
        ip_info.get("city", "Unknown"),
        ip_info.get("region", "Unknown"),
        ip_info.get("country", "Unknown"),
        ip_info.get("country_code", "XX"),
        ip_info.get("org", "Unknown"),
        ip_info.get("hostname", ""),
        ip_info.get("is_business", False),
        visitor.current_url,
        visitor.referrer,
        json.dumps(visitor.pages_viewed),
        visitor.visit_duration,
        visitor.user_agent,
        visitor.screen_size,
        enriched,
        lead_id
    ))
    
    conn.commit()
    conn.close()

async def process_visitor(visitor_data: VisitorData):
    """Process a visitor and create/update lead"""
    global settings
    
    # Step 0: Get IP information first
    ip_info = await get_ip_info(visitor_data.ip_address or "")
    
    # Extract domain from URL or IP
    domain = extract_domain(visitor_data.current_url)
    if not domain or domain in ["localhost", "127.0.0.1"]:
        # For demo, use a test domain
        domain = "notion.so"
    
    lead_id = f"lead_{int(datetime.now().timestamp())}_{visitor_data.session_id[:8]}"
    enriched = False
    
    # Step 1: Enrich company (only if business IP or always for demo)
    company_data = {}
    if settings.get("apollo_api_key"):
        result = await enrich_company_apollo(domain, settings["apollo_api_key"])
        if result.get("success"):
            company_data = result["data"]
            enriched = True
    
    if not company_data:
        company_data = {
            "name": domain.split(".")[0].title(),
            "domain": domain,
            "industry": "Unknown",
            "employee_count": 0,
            "country": "Unknown"
        }
    
    # Step 2: Find contacts
    contacts = []
    if settings.get("hunter_api_key"):
        result = await find_contacts_hunter(domain, settings["hunter_api_key"])
        if result.get("success"):
            contacts = result["contacts"]
    
    # Step 3: Score with AI
    scoring_result = {"data": {
        "icp_score": 50,
        "tier": "warm",
        "qualified": False,
        "match_reasons": [],
        "miss_reasons": ["AI scoring not configured"],
        "intent_signals": [],
        "recommended_action": "send_email",
        "urgency": "medium",
        "research_summary": "Lead pending AI analysis",
        "talking_points": [],
        "email_draft": {}
    }}
    
    if settings.get("openai_api_key"):
        visit_data = {
            "pages_viewed": visitor_data.pages_viewed,
            "visit_duration": visitor_data.visit_duration,
            "referrer": visitor_data.referrer
        }
        result = await score_lead_openai(
            company_data, 
            contacts, 
            visit_data, 
            settings["openai_api_key"],
            settings.get("icp_config", {})
        )
        if result.get("success"):
            scoring_result = result
    
    # Step 4: Save visitor to database (ALL visitors)
    save_visitor_to_db(visitor_data.session_id, ip_info, visitor_data, enriched, lead_id if enriched else None)
    
    # Step 5: Save lead to database (only enriched leads)
    if enriched:
        save_lead_to_db(lead_id, company_data, contacts, visitor_data, scoring_result["data"])
    
    # Step 6: Send Slack notification (if configured and hot lead)
    if settings.get("slack_webhook_url") and scoring_result["data"].get("tier") == "hot":
        await send_slack_notification(company_data, contacts, scoring_result["data"])
    
    return {
        "lead_id": lead_id if enriched else None,
        "visitor": {
            "session_id": visitor_data.session_id,
            "ip_info": ip_info,
            "enriched": enriched
        },
        "company": company_data if enriched else None,
        "contacts": contacts,
        "scoring": scoring_result["data"] if enriched else None
    }

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.replace("www.", "")
        if ":" in domain:
            domain = domain.split(":")[0]
        return domain
    except:
        return ""

def save_lead_to_db(lead_id: str, company: Dict, contacts: List, visitor: VisitorData, scoring: Dict):
    """Save lead data to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Insert/update company
        cursor.execute("""
            INSERT OR REPLACE INTO companies 
            (domain, name, industry, employee_count, country, city, description, 
             funding_stage, total_funding, annual_revenue, website, linkedin_url, enriched_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            company.get("domain", ""),
            company.get("name", ""),
            company.get("industry", ""),
            company.get("employee_count", 0),
            company.get("country", ""),
            company.get("city", ""),
            company.get("description", ""),
            company.get("funding_stage", ""),
            company.get("total_funding", 0),
            company.get("annual_revenue", 0),
            company.get("website", ""),
            company.get("linkedin_url", ""),
            json.dumps(company)
        ))
        
        company_id = cursor.lastrowid or cursor.execute(
            "SELECT id FROM companies WHERE domain = ?", (company.get("domain", ""),)
        ).fetchone()[0]
        
        # Insert contacts
        for contact in contacts:
            cursor.execute("""
                INSERT OR IGNORE INTO contacts 
                (company_id, name, first_name, last_name, email, title, seniority, 
                 department, linkedin_url, phone, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company_id,
                contact.get("name", ""),
                contact.get("first_name", ""),
                contact.get("last_name", ""),
                contact.get("email", ""),
                contact.get("title", ""),
                contact.get("seniority", ""),
                contact.get("department", ""),
                contact.get("linkedin_url", ""),
                contact.get("phone", ""),
                contact.get("confidence", 0)
            ))
        
        # Insert lead
        best_contact = scoring.get("best_contact", {})
        cursor.execute("""
            INSERT INTO leads 
            (lead_id, company_id, session_id, ip_address, pages_viewed, visit_duration,
             referrer, user_agent, icp_score, intent_score, tier, qualified,
             match_reasons, miss_reasons, intent_signals, research_summary,
             talking_points, recommended_action, urgency, email_draft, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lead_id,
            company_id,
            visitor.session_id,
            visitor.ip_address,
            json.dumps(visitor.pages_viewed),
            visitor.visit_duration,
            visitor.referrer,
            visitor.user_agent,
            scoring.get("icp_score", 0),
            scoring.get("intent_score", 0),
            scoring.get("tier", "cold"),
            scoring.get("qualified", False),
            json.dumps(scoring.get("match_reasons", [])),
            json.dumps(scoring.get("miss_reasons", [])),
            json.dumps(scoring.get("intent_signals", [])),
            scoring.get("research_summary", ""),
            json.dumps(scoring.get("talking_points", [])),
            scoring.get("recommended_action", ""),
            scoring.get("urgency", "low"),
            json.dumps(scoring.get("email_draft", {})),
            "new"
        ))
        
        conn.commit()
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

async def send_slack_notification(company: Dict, contacts: List, scoring: Dict):
    """Send Slack notification for hot leads"""
    if not settings.get("slack_webhook_url"):
        return
    
    best_contact = scoring.get("best_contact", {})
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔥 HOT LEAD ALERT"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Company:*\n{company.get('name', 'Unknown')}"},
                    {"type": "mrkdwn", "text": f"*Score:*\n{scoring.get('icp_score', 0)}/100"},
                    {"type": "mrkdwn", "text": f"*Industry:*\n{company.get('industry', 'Unknown')}"},
                    {"type": "mrkdwn", "text": f"*Contact:*\n{best_contact.get('name', 'N/A')}"}
                ]
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Research Summary:*\n{scoring.get('research_summary', '')}"}
            }
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            await client.post(settings["slack_webhook_url"], json=message)
        except:
            pass

# ============== API ENDPOINTS ==============

@app.on_event("startup")
async def startup():
    init_db()
    load_settings()

@app.get("/")
async def root():
    return {"message": "Swan AI Clone API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Webhook for tracking script
@app.post("/webhook/visitor")
async def receive_visitor(visitor: VisitorData, request: Request, background_tasks: BackgroundTasks):
    """Receive visitor data from tracking script"""
    # Capture real IP from request headers
    real_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
        request.headers.get("X-Real-IP", "") or
        request.headers.get("CF-Connecting-IP", "") or  # Cloudflare
        (request.client.host if request.client else "")
    )
    
    # Override the IP in visitor data with the real IP
    visitor.ip_address = real_ip
    
    print(f"🦢 Visitor tracked: IP={real_ip}, Session={visitor.session_id}, URL={visitor.current_url}")
    
    background_tasks.add_task(process_visitor, visitor)
    return {"status": "queued", "session_id": visitor.session_id, "ip": real_ip}

# Test endpoint for manual testing
@app.post("/api/test-visitor")
async def test_visitor(request: Request, domain: str = "notion.so"):
    """Test endpoint to simulate a visitor"""
    # Get the IP of the person testing
    real_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
        request.headers.get("X-Real-IP", "") or
        (request.client.host if request.client else "127.0.0.1")
    )
    
    visitor = VisitorData(
        project_id="test_project",
        session_id=f"test_{int(datetime.now().timestamp())}",
        ip_address=real_ip,
        current_url=f"https://{domain}/pricing",
        referrer="https://google.com",
        pages_viewed=[
            {"url": "/", "time": datetime.now().isoformat()},
            {"url": "/features", "time": datetime.now().isoformat()},
            {"url": "/pricing", "time": datetime.now().isoformat()}
        ],
        visit_duration=180,
        timestamp=datetime.now().isoformat()
    )
    result = await process_visitor(visitor)
    return result

# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    """Get current settings (without exposing full keys)"""
    return {
        "apollo_api_key": "***" + settings.get("apollo_api_key", "")[-4:] if settings.get("apollo_api_key") else "",
        "hunter_api_key": "***" + settings.get("hunter_api_key", "")[-4:] if settings.get("hunter_api_key") else "",
        "openai_api_key": "***" + settings.get("openai_api_key", "")[-4:] if settings.get("openai_api_key") else "",
        "slack_webhook_url": "configured" if settings.get("slack_webhook_url") else "",
        "icp_config": settings.get("icp_config", {})
    }

@app.post("/api/settings")
async def update_settings(new_settings: APISettings):
    """Update API settings"""
    global settings
    if new_settings.apollo_api_key:
        settings["apollo_api_key"] = new_settings.apollo_api_key
        save_setting("apollo_api_key", new_settings.apollo_api_key)
    if new_settings.hunter_api_key:
        settings["hunter_api_key"] = new_settings.hunter_api_key
        save_setting("hunter_api_key", new_settings.hunter_api_key)
    if new_settings.openai_api_key:
        settings["openai_api_key"] = new_settings.openai_api_key
        save_setting("openai_api_key", new_settings.openai_api_key)
    if new_settings.slack_webhook_url:
        settings["slack_webhook_url"] = new_settings.slack_webhook_url
        save_setting("slack_webhook_url", new_settings.slack_webhook_url)
    return {"status": "updated"}

@app.post("/api/settings/icp")
async def update_icp(icp: ICPConfig):
    """Update ICP configuration"""
    global settings
    settings["icp_config"] = icp.dict()
    save_setting("icp_config", icp.dict())
    return {"status": "updated"}

# Test API connections
@app.post("/api/test-api")
async def test_api(request: TestAPIRequest):
    """Test an API connection"""
    if request.api_type == "ipinfo":
        # Test IPInfo API
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                test_ip = "8.8.8.8"  # Google DNS
                url = f"https://ipinfo.io/{test_ip}/json?token={request.api_key}"
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True, 
                        "message": "IPInfo API connected!",
                        "data": {
                            "ip": data.get("ip"),
                            "city": data.get("city"),
                            "region": data.get("region"),
                            "country": data.get("country"),
                            "org": data.get("org")
                        }
                    }
                else:
                    return {"success": False, "error": f"API error: {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
    elif request.api_type == "apollo":
        result = await enrich_company_apollo(request.test_domain, request.api_key)
        return result
    elif request.api_type == "hunter":
        result = await find_contacts_hunter(request.test_domain, request.api_key)
        return result
    elif request.api_type == "openai":
        # Simple test
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {request.api_key}"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "Say 'API working!' in 3 words"}],
                        "max_tokens": 20
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    reply = data["choices"][0]["message"]["content"]
                    return {"success": True, "message": "OpenAI API connected!", "data": {"response": reply}}
                else:
                    return {"success": False, "error": response.text}
            except Exception as e:
                return {"success": False, "error": str(e)}
    elif request.api_type == "slack":
        # Test Slack webhook
        if not request.api_key.startswith("https://hooks.slack.com/"):
            return {"success": False, "error": "Invalid Slack webhook URL"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    request.api_key,
                    json={"text": "🦢 Swan AI Clone - Test notification! Your webhook is working."}
                )
                if response.status_code == 200:
                    return {"success": True, "message": "Slack webhook working! Check your channel."}
                else:
                    return {"success": False, "error": f"Webhook error: {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
    else:
        return {"error": "Unknown API type"}

# Leads endpoints
# Visitors endpoints
@app.get("/api/visitors")
async def get_visitors(limit: int = 100):
    """Get all visitors with IP info"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM visitors 
        ORDER BY updated_at DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    
    visitors = []
    for row in rows:
        visitor = dict(row)
        # Parse pages_viewed JSON
        if visitor.get("pages_viewed"):
            try:
                visitor["pages_viewed"] = json.loads(visitor["pages_viewed"])
            except:
                visitor["pages_viewed"] = []
        visitors.append(visitor)
    
    conn.close()
    return {"visitors": visitors, "total": len(visitors)}

@app.get("/api/visitors/{session_id}")
async def get_visitor(session_id: str):
    """Get single visitor details"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM visitors WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Visitor not found")
    
    visitor = dict(row)
    if visitor.get("pages_viewed"):
        try:
            visitor["pages_viewed"] = json.loads(visitor["pages_viewed"])
        except:
            visitor["pages_viewed"] = []
    
    conn.close()
    return visitor

@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total visitors
    cursor.execute("SELECT COUNT(*) FROM visitors")
    total_visitors = cursor.fetchone()[0]
    
    # Business visitors
    cursor.execute("SELECT COUNT(*) FROM visitors WHERE is_business = 1")
    business_visitors = cursor.fetchone()[0]
    
    # Enriched leads
    cursor.execute("SELECT COUNT(*) FROM leads")
    total_leads = cursor.fetchone()[0]
    
    # Hot leads
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'hot'")
    hot_leads = cursor.fetchone()[0]
    
    # Warm leads
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'warm'")
    warm_leads = cursor.fetchone()[0]
    
    # Cold leads
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'cold'")
    cold_leads = cursor.fetchone()[0]
    
    # Average score
    cursor.execute("SELECT AVG(icp_score) FROM leads")
    avg_score = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "total_visitors": total_visitors,
        "business_visitors": business_visitors,
        "total_leads": total_leads,
        "hot_leads": hot_leads,
        "warm_leads": warm_leads,
        "cold_leads": cold_leads,
        "avg_score": round(avg_score, 1)
    }

# Leads endpoints
@app.get("/api/leads")
async def get_leads(limit: int = 50, tier: Optional[str] = None):
    """Get all leads"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT l.*, c.name as company_name, c.domain, c.industry, c.employee_count,
               c.country, c.funding_stage, c.total_funding, c.description
        FROM leads l
        LEFT JOIN companies c ON l.company_id = c.id
    """
    if tier:
        query += f" WHERE l.tier = '{tier}'"
    query += " ORDER BY l.created_at DESC LIMIT ?"
    
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    
    leads = []
    for row in rows:
        lead = dict(row)
        # Parse JSON fields
        for field in ['pages_viewed', 'match_reasons', 'miss_reasons', 'intent_signals', 'talking_points', 'email_draft']:
            if lead.get(field):
                try:
                    lead[field] = json.loads(lead[field])
                except:
                    pass
        leads.append(lead)
    
    conn.close()
    return {"leads": leads, "total": len(leads)}

@app.get("/api/leads/{lead_id}")
async def get_lead(lead_id: str):
    """Get single lead with contacts"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get lead
    cursor.execute("""
        SELECT l.*, c.name as company_name, c.domain, c.industry, c.employee_count,
               c.country, c.city, c.funding_stage, c.total_funding, c.annual_revenue,
               c.description, c.website, c.linkedin_url
        FROM leads l
        LEFT JOIN companies c ON l.company_id = c.id
        WHERE l.lead_id = ?
    """, (lead_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    lead = dict(row)
    
    # Parse JSON fields
    for field in ['pages_viewed', 'match_reasons', 'miss_reasons', 'intent_signals', 'talking_points', 'email_draft']:
        if lead.get(field):
            try:
                lead[field] = json.loads(lead[field])
            except:
                pass
    
    # Get contacts
    cursor.execute("""
        SELECT * FROM contacts WHERE company_id = ?
    """, (lead.get("company_id"),))
    contacts = [dict(row) for row in cursor.fetchall()]
    lead["contacts"] = contacts
    
    conn.close()
    return lead

@app.delete("/api/leads/{lead_id}")
async def delete_lead(lead_id: str):
    """Delete a lead"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leads WHERE lead_id = ?", (lead_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM leads")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'hot'")
    hot = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'warm'")
    warm = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM leads WHERE tier = 'cold'")
    cold = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(icp_score) FROM leads")
    avg_score = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "total_leads": total,
        "hot_leads": hot,
        "warm_leads": warm,
        "cold_leads": cold,
        "avg_score": round(avg_score, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
