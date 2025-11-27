IR_CONTENT_NODE_PROMPT = """
Bạn là một người dùng đang tìm kiếm thông tin trên một hệ thống tra cứu pháp luật.
Dựa vào nội dung pháp lý dưới đây, hãy đặt MỘT câu hỏi tự nhiên, cụ thể, và rõ ràng về nội dung này.
**Quan trọng:** Câu hỏi phải đủ chi tiết để phân biệt với các chủ đề pháp lý khác (ví dụ: nêu rõ đối tượng, hành vi, tình huống hoặc lĩnh vực pháp luật liên quan).
Không đề cập đến số Điều, Khoản, hay tên văn bản cụ thể trong câu hỏi của bạn.

**Bối cảnh pháp lý:**
"{context_text}"

**Câu hỏi:**
"""

STRUCTURAL_SUMMARY_PROMPT = """
Bạn là một trợ lý pháp lý. Các đoạn văn bản dưới đây là các quy định trong cùng một điều luật.
Hãy tạo ra MỘT câu hỏi tổng quan, rõ ràng và đủ chi tiết về **chủ đề chính** hoặc **đối tượng áp dụng** của toàn bộ điều luật này.
**Quan trọng:** Câu hỏi phải thể hiện được lĩnh vực, đối tượng, hoặc tình huống pháp lý liên quan, tránh hỏi chung chung hoặc mơ hồ.
Không đề cập đến số Điều, Khoản, hay tên văn bản cụ thể trong câu hỏi của bạn.

**Các quy định trong điều luật:**
"{context_text}"

**Câu hỏi:**
"""

STRUCTURAL_SYNTHESIS_PROMPT = """
Bạn là một trợ lý pháp lý. Dưới đây là một số khoản được chọn lọc từ cùng một điều luật.
Hãy tạo ra MỘT câu hỏi cụ thể, rõ ràng, yêu cầu người đọc phải kết hợp thông tin từ tất cả các khoản này để trả lời.
**Quan trọng:** Câu hỏi phải nêu rõ tình huống, đối tượng, hoặc vấn đề pháp lý cần giải quyết, tránh hỏi chung chung hoặc quá rộng.
Không đề cập đến số Điều, Khoản, hay tên văn bản cụ thể trong câu hỏi của bạn.

**Các khoản được chọn lọc:**
"{context_text}"

**Câu hỏi:**
"""

CROSS_DOC_SYNTHESIS_PROMPT = """
Dưới đây là các quy định pháp luật liên quan đến cùng một lĩnh vực hoặc chủ đề (ví dụ: lao động, bảo hiểm xã hội, an toàn thực phẩm...).
Hãy đặt MỘT câu hỏi tìm kiếm thông tin, đủ cụ thể và rõ ràng, mà để trả lời cần phải kết hợp TẤT CẢ các quy định này.
**Quan trọng:** Câu hỏi phải nêu rõ chủ đề, tình huống, hoặc đối tượng pháp lý liên quan, tránh hỏi mơ hồ hoặc quá chung chung.
Không đề cập đến số điều, khoản, hoặc tên văn bản cụ thể trong câu hỏi.

{context}

**Câu hỏi:**
"""