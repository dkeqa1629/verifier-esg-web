import os
import json
import re
from http.server import BaseHTTPRequestHandler


# === ESG Database (Mock) ===
class TrustedESGDatabase:
    def __init__(self):
        self._mock_data = {
            'AAPL': {'esg_risk': 16.7, 'controversy_level': 2},
            'MSFT': {'esg_risk': 15.0, 'controversy_level': 1},
            'XOM':  {'esg_risk': 36.5, 'controversy_level': 4},
            'TSLA': {'esg_risk': 28.5, 'controversy_level': 3},
            'AMZN': {'esg_risk': 30.2, 'controversy_level': 3},
            'GOOGL': {'esg_risk': 24.1, 'controversy_level': 2},
            'META': {'esg_risk': 32.0, 'controversy_level': 4},
            'NVDA': {'esg_risk': 13.5, 'controversy_level': 1},
            'JPM':  {'esg_risk': 22.3, 'controversy_level': 2},
            'JNJ':  {'esg_risk': 18.9, 'controversy_level': 1},
            'VNM':  {'esg_risk': 35.2, 'controversy_level': 3},
            'V':    {'esg_risk': 14.2, 'controversy_level': 1},
            'WMT':  {'esg_risk': 27.8, 'controversy_level': 2},
            'DIS':  {'esg_risk': 20.5, 'controversy_level': 2},
            'NFLX': {'esg_risk': 19.3, 'controversy_level': 1},
        }

    def get_esg_data(self, ticker: str) -> dict:
        return self._mock_data.get(ticker.upper(), {'esg_risk': 45.0, 'controversy_level': 3})


# === Helper Functions ===
def extract_json(text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except json.JSONDecodeError:
        return {"Action": "Error", "Reasoning": "Cannot parse LLM output"}


def get_real_time_price(ticker: str) -> float:
    """Lấy giá thị trường thực tế bằng yfinance"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if not todays_data.empty:
            return round(todays_data['Close'].iloc[0], 2)
    except Exception:
        pass
    # Fallback prices
    fallback = {
        'AAPL': 198.50, 'MSFT': 420.30, 'GOOGL': 175.20, 'NVDA': 880.50,
        'TSLA': 248.90, 'AMZN': 186.40, 'META': 505.70, 'JPM': 198.60,
        'JNJ': 155.40, 'VNM': 4.50, 'V': 280.30, 'WMT': 165.80,
        'DIS': 112.40, 'NFLX': 628.90, 'XOM': 104.50
    }
    return fallback.get(ticker.upper(), 150.0)


def parse_user_prompt(message: str, api_key: str) -> list:
    """Dùng Gemini để trích xuất mã chứng khoán từ prompt người dùng"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    sys_prompt = """Bạn là một trợ lý trích xuất mã chứng khoán. 
    Nhiệm vụ: tìm ra TẤT CẢ các mã chứng khoán được nhắc đến và trả về JSON Array.
    Ví dụ: "Hãy phân tích cho tôi con AAPL với Microsoft" -> ["AAPL", "MSFT"]
    Nếu người dùng chỉ chào hỏi hoặc không nhắc mã cổ phiếu nào, trả về [].
    Không giải thích, chỉ trả về JSON Array."""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=sys_prompt + "\n\nYêu cầu của người dùng: " + message,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    try:
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(response.text)
    except Exception:
        return []


def llm_decide(ticker: str, esg_data: dict, price: float, budget: float, api_key: str, error_msg: str = "") -> dict:
    """Gemini ra quyết định đầu tư"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = f"""Bạn là một chuyên gia quản lý quỹ ESG kiêm Chuyên viên Định giá. Hãy ra quyết định cho cổ phiếu {ticker}.
- Dữ liệu nội bộ: Rủi ro ESG={esg_data['esg_risk']}, Mức độ tai tiếng={esg_data['controversy_level']}.
- Dữ liệu thị trường: Giá cổ phiếu hiện tại là {price}$.
- Ngân sách hiện có: {budget}$.
- LUẬT: Nếu ESG Risk > 30 hoặc Controversy > 2, BẮT BUỘC phải chọn Bán hoặc Giữ.
"""
    if error_msg:
        prompt += f"\n[CẢNH BÁO TỪ Z3]: Quyết định trước bị TỪ CHỐI vì: '{error_msg}'. BẮT BUỘC sửa thông số.\n"

    prompt += """\nTrình bày quyết định DƯỚI DẠNG JSON:
{
    "Company_Name": "Tên mã",
    "Action": "Mua" hoặc "Bán" hoặc "Giữ",
    "Quantity": (Tính toán số lượng cổ phiếu hợp lý dựa trên giá và ngân sách),
    "Target_Price": (Mức giá mục tiêu),
    "Reasoning": "Lý do"
}"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return extract_json(response.text)


def z3_verify(decision: dict, esg_data: dict, budget: float) -> tuple:
    """Z3 Guardrail: Kiểm tra ESG + Tài chính"""
    from z3 import Solver, Real, Int, Bool, Implies, sat

    if decision.get("Action") == "Error":
        return False, "LLM Error"

    solver = Solver()
    esg_risk_z3 = Real('esg_risk_z3')
    controversy_z3 = Int('controversy_z3')
    is_buy = Bool('is_buy')
    quantity_z3 = Real('quantity_z3')
    target_price_z3 = Real('target_price_z3')
    budget_z3 = Real('budget_z3')

    solver.add(esg_risk_z3 == esg_data['esg_risk'])
    solver.add(controversy_z3 == esg_data['controversy_level'])
    solver.add(budget_z3 == budget)

    action = str(decision.get("Action", "")).upper()
    try:
        qty = float(decision.get("Quantity", 0))
        price = float(decision.get("Target_Price", 0.0))
    except (ValueError, TypeError):
        qty, price = 0.0, 0.0

    solver.add(quantity_z3 == qty)
    solver.add(target_price_z3 == price)

    if action == "MUA":
        solver.add(is_buy == True)
    else:
        solver.add(is_buy == False)

    # LUẬT 1: ESG
    solver.add(Implies(is_buy, esg_risk_z3 <= 30.0))
    solver.add(Implies(is_buy, controversy_z3 <= 2))
    # LUẬT 2: Ngân sách
    solver.add(Implies(is_buy, (quantity_z3 * target_price_z3) <= budget_z3))

    if solver.check() == sat:
        return True, "Passed"
    else:
        if qty * price > budget:
            return False, f"Vi phạm Ngân sách! ({qty} x {price} = {qty*price}$) > {budget}$"
        else:
            return False, f"Vi phạm ESG! Risk={esg_data['esg_risk']}, Controversy={esg_data['controversy_level']}"


def generate_greeting(message: str, api_key: str) -> str:
    """Generate a greeting response when no stock tickers are found"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = f"""Bạn là Trợ lý AI Đầu tư ESG của Verifier.AI. Bạn giúp nhà đầu tư kiểm định danh mục theo tiêu chuẩn ESG khắt khe nhất, sử dụng hệ thống Neuro-Symbolic (Gemini + Z3 SMT Solver).

Người dùng vừa nói: "{message}"

Hãy trả lời ngắn gọn, thân thiện bằng tiếng Việt. Nếu họ chào hỏi, hãy chào lại và giới thiệu ngắn về khả năng của bạn. Gợi ý họ thử nhập lệnh như: "Kiểm tra cho tôi AAPL, NVDA" hoặc "Tôi có 10.000$, hãy mua MSFT và GOOGL"."""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text


# === Main Vercel Handler ===
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        message = data.get('message', '')
        budget = float(data.get('budget', 10000))
        api_key = os.environ.get('GEMINI_API_KEY', '')

        if not api_key:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(
                {"error": "GEMINI_API_KEY not configured"}).encode())
            return

        try:
            # Step 1: Parse tickers from user message
            tickers = parse_user_prompt(message, api_key)

            if not tickers:
                # No tickers found - generate a greeting/help response
                greeting = generate_greeting(message, api_key)
                result = {
                    "type": "greeting",
                    "message": greeting,
                    "tickers": []
                }
            else:
                # Step 2: Process each ticker
                esg_db = TrustedESGDatabase()
                results = []

                for ticker in tickers:
                    ticker = ticker.upper()
                    esg_data = esg_db.get_esg_data(ticker)
                    market_price = get_real_time_price(ticker)

                    # LLM + Z3 verification loop (max 3 attempts)
                    error_msg = ""
                    final_result = None

                    for attempt in range(1, 4):
                        decision = llm_decide(
                            ticker, esg_data, market_price, budget, api_key, error_msg)
                        is_safe, msg = z3_verify(decision, esg_data, budget)

                        if is_safe:
                            final_result = {
                                "ticker": ticker,
                                "status": "approved",
                                "decision": decision,
                                "market_price": market_price,
                                "esg_data": esg_data,
                                "z3_message": "Passed",
                                "attempts": attempt
                            }
                            break
                        else:
                            error_msg = msg
                            if attempt == 3:
                                final_result = {
                                    "ticker": ticker,
                                    "status": "rejected",
                                    "decision": decision,
                                    "market_price": market_price,
                                    "esg_data": esg_data,
                                    "z3_message": msg,
                                    "attempts": attempt
                                }

                    results.append(final_result)

                result = {
                    "type": "analysis",
                    "message": f"Đã phân tích {len(tickers)} mã chứng khoán.",
                    "tickers": tickers,
                    "results": results,
                    "budget": budget
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
