"""Interactive terminal client for FinQuery RAG system.

Run locally on Windows:
    python scripts/chat_cli.py

Features:
- Automatic JWT authentication
- Maintains multi-turn conversation session
- Formatted output for:
  * Regular authoritative answers + citations
  * Interactive clarification prompts with numbered options
  * Cell coordinates and PDF visual bounding boxes
  * Deterministic calculation results (DuPont, Margins, CAGR)
"""

import sys
import time
import uuid
import requests

# Ensure console handles UTF-8 cleanly on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:18002"

# Color helpers
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_auth_token():
    """Login or register test user to obtain JWT token."""
    credentials = {"email": "analyst@nano-finrag.internal", "password": "analyst-pass-123456"}
    try:
        r = requests.post(f"{BASE_URL}/login", json=credentials, timeout=5)
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception:
        pass

    try:
        r = requests.post(f"{BASE_URL}/register", json=credentials, timeout=5)
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception as exc:
        print(f"{Colors.RED}[!] Failed to authenticate: {exc}{Colors.END}")
        return None
    return None


def print_banner():
    print(f"\n{Colors.BOLD}{Colors.CYAN}======================================================================{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}       Nano_finRAG 机构级智能财报问答与多轮消歧交互终端 (CLI)       {Colors.END}")
    print(f"{Colors.CYAN}  后端地址: {BASE_URL}  |  Fail-Closed 0.00% 形式化放行保障  {Colors.END}")
    print(f"{Colors.CYAN}======================================================================{Colors.END}")
    print(f"{Colors.YELLOW}推荐测试场景示例:{Colors.END}")
    print(f"  1. 交互消歧测试: {Colors.BOLD}According to Apple's FY2025 10-K filing, what was the reported Deferred?{Colors.END}")
    print(f"  2. 选项匹配测试: {Colors.BOLD}包含347的那项{Colors.END}  或  {Colors.BOLD}第3项{Colors.END}  或  {Colors.BOLD}604{Colors.END}")
    print(f"  3. 历史撤销重选: {Colors.BOLD}不对，换成第1项{Colors.END}")
    print(f"  4. 综合财务查询: {Colors.BOLD}Compare the gross margin of Apple and Microsoft in FY2024{Colors.END}")
    print(f"  5. 市场行情查询: {Colors.BOLD}比亚迪最近股价怎么样{Colors.END}  或  {Colors.BOLD}Apple stock price{Colors.END}")
    print(f"  输入 {Colors.BOLD}'exit'{Colors.END} 或 {Colors.BOLD}'quit'{Colors.END} 退出程序，输入 {Colors.BOLD}'reset'{Colors.END} 清空当前会话上下文。\n")


def main():
    token = get_auth_token()
    if not token:
        print(f"{Colors.RED}无法连接后端或登录失败，请确认服务已启动并在 18002 端口监听。{Colors.END}")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}
    session_id = f"cli_{uuid.uuid4().hex[:12]}"
    turn_count = 1

    print_banner()

    while True:
        try:
            prompt_label = f"{Colors.BOLD}{Colors.BLUE}[Turn {turn_count}] 请输入您的问题 > {Colors.END}"
            user_input = input(prompt_label).strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print(f"{Colors.CYAN}会话已结束。感谢使用！{Colors.END}")
                break

            if user_input.lower() == "reset":
                session_id = f"cli_{uuid.uuid4().hex[:12]}"
                turn_count = 1
                print(f"{Colors.YELLOW}[*] 已重置会话，开启全新的上下文。{Colors.END}\n")
                continue

            start_t = time.time()
            print(f"{Colors.YELLOW}正在由 GPU 2 专家模型与形式化计算沙盒处理中...{Colors.END}", end="\r")

            resp = requests.post(
                f"{BASE_URL}/query",
                json={"question": user_input, "session_id": session_id},
                headers=headers,
                timeout=60,
            )
            elapsed = time.time() - start_t

            if resp.status_code == 401:
                # Token expired or service restarted; automatically refresh credentials and retry once
                new_token = get_auth_token()
                if new_token:
                    token = new_token
                    headers = {"Authorization": f"Bearer {token}"}
                    resp = requests.post(
                        f"{BASE_URL}/query",
                        json={"question": user_input, "session_id": session_id},
                        headers=headers,
                        timeout=60,
                    )
                    elapsed = time.time() - start_t

            if resp.status_code != 200:
                print(f"\n{Colors.RED}[!] 请求失败 (HTTP {resp.status_code}): {resp.text}{Colors.END}\n")
                continue

            data = resp.json()
            status = data.get("status") or "SUCCESS"
            answer = data.get("answer", "")
            rewritten = data.get("rewritten_question")
            sources = data.get("sources", [])
            clarification = data.get("clarification")

            print(" " * 60, end="\r")  # Clear waiting message

            # Case A: Clarification Required
            if status == "CLARIFICATION_REQUIRED":
                opts = clarification.get("options", []) if clarification else []
                if opts:
                    print(f"{Colors.BOLD}{Colors.YELLOW}[!] 【系统主动反思追问】检测到报表存在多项重名/不同口径披露:{Colors.END}")
                    print(f"{Colors.YELLOW}{answer}{Colors.END}\n")
                    print(f"{Colors.BOLD}系统推荐候选选项列表:{Colors.END}")
                    for idx, opt in enumerate(opts, 1):
                        if isinstance(opt, dict):
                            val = opt.get("target_value") or opt.get("label") or opt.get("value")
                            desc = opt.get("preview_text") or opt.get("description") or opt.get("context")
                            page = opt.get("page")
                            opt_id = opt.get("option_id", idx)
                            page_str = f" (第 {page} 页)" if page is not None else ""
                            desc_str = f" - {desc}" if desc else ""
                            print(f"  * {Colors.BOLD}[选项 {opt_id}]{Colors.END} 目标值: {Colors.GREEN}{val}{Colors.END}{page_str}{desc_str}")
                        else:
                            print(f"  * {Colors.GREEN}{opt}{Colors.END}")
                    print(f"\n{Colors.CYAN}提示: 您可以在下一轮直接输入序号（如 '第3项' 或 '3'）、具体数值（如 '604'）或所在行特征关键词（如 '包含347的那项'）。{Colors.END}\n")
                else:
                    print(f"{Colors.BOLD}{Colors.YELLOW}[!] 【系统提示：需要补充查询上下文】{Colors.END}")
                    print(f"{Colors.YELLOW}{answer}{Colors.END}\n")
                    print(f"{Colors.CYAN}提示: 您的提问缺少具体的目标公司、财务科目或财报年份。请输入例如 'According to Apple\\'s FY2025 10-K, what was Deferred?' 的完整财务问题。{Colors.END}\n")

            # Case B: Refusal / Fail-Closed / Service Error
            elif status in ("FAIL_CLOSED", "ERROR"):
                print(f"{Colors.BOLD}{Colors.RED}[X] 【未达放行标准 (Fail-Closed 形式化拦截)】 (耗时 {elapsed:.2f}s):{Colors.END}")
                if rewritten and rewritten != user_input:
                    print(f"   {Colors.CYAN}多轮重构独立查询: {rewritten}{Colors.END}")
                print(f"\n   {Colors.RED}{answer}{Colors.END}\n")
                print(f"   {Colors.YELLOW}说明: 底层审计底稿缺乏充分证据、指标未在此财报中明确披露或上游大模型规划服务受限，系统执行 0.00% 形式化防幻觉阻断。{Colors.END}\n")

            # Case C: Out of Scope
            elif status == "OUT_OF_SCOPE":
                print(f"{Colors.BOLD}{Colors.RED}[-] 【超出财报支持范围 (Out-of-Scope)】:{Colors.END}")
                print(f"\n   {Colors.RED}{answer}{Colors.END}\n")
                print(f"   {Colors.YELLOW}说明: 该问题不属于系统支持的上市公司财务报表分析范围。{Colors.END}\n")

            # Case D1: Conversational / Interactive Guidance
            elif data.get("intent") == "conversation":
                print(f"{Colors.BOLD}{Colors.CYAN}[i] 【系统提示 / 交互说明】 (耗时 {elapsed:.2f}s):{Colors.END}")
                print(f"\n   {Colors.CYAN}{answer}{Colors.END}\n")

            # Case D2: Market Quote Released
            elif data.get("intent") == "market_quote":
                print(f"{Colors.BOLD}{Colors.GREEN}[📈 实时市场行情放行 (Market API)] (耗时 {elapsed:.2f}s):{Colors.END}")
                print(f"\n   {Colors.GREEN}{answer}{Colors.END}\n")

            # Case D3: Authoritative Answer Released
            else:
                print(f"{Colors.BOLD}{Colors.GREEN}[√] 【权威已放行答复】 (耗时 {elapsed:.2f}s):{Colors.END}")
                if rewritten and rewritten != user_input:
                    print(f"   {Colors.CYAN}多轮重构独立查询: {rewritten}{Colors.END}")
                print(f"\n   {Colors.BOLD}{answer}{Colors.END}\n")

                calcs = data.get("calculations", [])
                if calcs:
                    print(f"   {Colors.BOLD}【确定性财务公式计算沙盒】:{Colors.END}")
                    for c in calcs:
                        op = c.get("operation") or c.get("op", "N/A")
                        val = c.get("value", "N/A")
                        formula = c.get("formula", "")
                        print(f"     • 操作: {op} | 计算结果: {Colors.GREEN}{val}{Colors.END} | 公式依据: {formula}")

                        # 自动解析双实体差额对比，生成通俗直观结论
                        operands = c.get("operands", [])
                        if op == "difference" and len(operands) == 2:
                            try:
                                v1 = float(operands[0].get("value", 0))
                                v2 = float(operands[1].get("value", 0))
                                d1 = operands[0].get("document_name", "").split("_")[0].upper()
                                d2 = operands[1].get("document_name", "").split("_")[0].upper()
                                entity_cn = {
                                    "AAPL": "苹果 (Apple)", "MSFT": "微软 (Microsoft)",
                                    "TSLA": "特斯拉 (Tesla)", "NVDA": "英伟达 (NVIDIA)",
                                    "AMZN": "亚马逊 (Amazon)", "GOOGL": "谷歌 (Google)",
                                    "JPM": "摩根大通 (JPMorgan)", "KO": "可口可乐 (Coca-Cola)"
                                }
                                name1 = entity_cn.get(d1, d1 or "主体 A")
                                name2 = entity_cn.get(d2, d2 or "主体 B")
                                if v1 > v2:
                                    winner, loser = name1, name2
                                    higher_val, lower_val = v1, v2
                                    diff = v1 - v2
                                elif v2 > v1:
                                    winner, loser = name2, name1
                                    higher_val, lower_val = v2, v1
                                    diff = v2 - v1
                                else:
                                    winner = None

                                if winner:
                                    print(f"\n   {Colors.BOLD}{Colors.YELLOW}【📊 研报分析直观结论】:{Colors.END}")
                                    print(f"     ⭐ {Colors.BOLD}{winner}{Colors.END} 的收益更高！")
                                    print(f"     • {winner}: {higher_val:,.2f} 百万美元")
                                    print(f"     • {loser}: {lower_val:,.2f} 百万美元")
                                    print(f"     • 领先优势: 高出 {Colors.GREEN}{diff:,.2f}{Colors.END} 百万美元")
                            except Exception:
                                pass
                    print()

                if sources:
                    print(f"   {Colors.BOLD}【审计底稿与证据链溯源】 (共 {len(sources)} 条权威引用):{Colors.END}")
                    for i, src in enumerate(sources, 1):
                        cite_id = src.get("citation_id", "N/A")
                        fn = src.get("filename", "10-K Filing")
                        p = src.get("page", "N/A")
                        ann = src.get("footnote_annotation")
                        coords = src.get("cell_coordinates")
                        bbox = src.get("bbox")

                        details = []
                        if coords:
                            details.append(f"单元格: Row {coords.get('row_idx')}, Col {coords.get('col_idx')}")
                        if bbox:
                            details.append(f"PDF红框Bbox: {bbox}")
                        if ann:
                            details.append(f"附注角标: {ann}")

                        extra_str = f" | {', '.join(details)}" if details else ""
                        print(f"     [{i}] {cite_id} | 文件: {fn} (第 {p} 页){extra_str}")
                print()

            turn_count += 1

        except KeyboardInterrupt:
            print(f"\n{Colors.CYAN}用户退出。{Colors.END}")
            break
        except Exception as exc:
            print(f"\n{Colors.RED}[!] 异常: {exc}{Colors.END}\n")


if __name__ == "__main__":
    main()
