# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đoàn Văn Tuấn
**Nhóm:** Nhóm 03 - E402
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity là điểm đánh giá độ tương đồng về mặt ngữ nghĩa giữa 2 câu, giá trị nằm trong khoảng [0, 1], càng gần 1 thì độ tương đồng càng cao và ngược lại. Sử dụng chỉ số này thì sẽ không quan tâm đến việc độ dài 2 câu có dài ngắn khác nhau, khác với việc sử dụng Euclidean Distance.

**Ví dụ HIGH similarity:**
- Sentence A: Tôi thích nuôi mèo.
- Sentence B: Lan rất yêu quý động vật, đặc biệt là chó con hoặc mèo vàng.
- Tại sao tương đồng: Có những động từ thể hiện sự yêu thích và cùng về yêu quý động vật còn cụ thể là mèo đều có trong cả 2 câu.

**Ví dụ LOW similarity:**
- Sentence A: Bầu trời hôm nay xanh thế.
- Sentence B: Con mèo nằm dưới gầm bàn.
- Tại sao khác: Nói về 2 vấn đề hoàn toàn khác biệt.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Đầu tiên là vấn đề độ dài dữ liệu, đây là vấn đề quan trong nhất. Cosine similarity chỉ quan tâm đến hướng, dù một câu ngắn hay một câu dài mà nói về cùng một chủ đề thì 2 vectors đó vẫn sẽ cùng hướng. Tuy nhiên nếu 2 câu có cùng chủ đề mà dài ngắn khác nhau có 1 cái là một câu, 1 cái khác làm một đoạn thì khoảng cách của chúng sẽ rất xa nhau nếu sử dụng Euclide Distance. 
> Thứ 2 là về tốc độ tính toán việc tính Cosine Similarity thực chất là phép tính tính vô hướng nếu 2 vectors này đều được chuẩn hóa về độ dài bằng 1.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Sau đoạn thêm đầu tiên,thì mỗi đoạn sau chỉ cần thêm 450 ký tự nữa là đủ một chunk. Phép tính: 500 + 450 * x >= 10.000 => x = 21.1 => cộng thêm chunk đầu 22.1 => Làm tròn 23
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap tăng lên thì số lượng chunk cũng tăng lên, muốn overlap để xác xuất chunk nắm giữ thông tin quan trọng sẽ nhiều hơn. Ví dụ khi ta chỉ cắt thông thường không overlap thì mỗi chunk chỉ mang dữ liệu của chunk đó nhưng nếu sử dụng overlap thì những chunk sau có thể mang cả dữ liệu của chunk trước đó xác suất mang thông tin quan trọng nhiều hơn.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Domain: Vietnamese Fairy Tales - Truyện cổ tích Việt Nam

**Tại sao nhóm chọn domain này?**
> Truyện cổ tích có context dài, nội dung phong phú và cấu trúc rõ ràng — phù hợp để kiểm tra khả năng retrieve chính xác của RAG 

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Sọ Dừa | https://loigiaihay.com/so-dua-truyen-co-tich-viet-nam-a185655.html | 5634 | story_title: ["Sọ Dừa"], story_type: "cổ tích", origin: "Việt Nam", themes: ["phép thuật", "tình yêu", "lòng tốt"], main_characters: ["Sọ Dừa", "Phú Ông", "cô út"]|
| 2 | Thạch Sanh | https://loigiaihay.com/thach-sanh-truyen-co-tich-viet-nam-a187017.html | 9207 | story_title: ["Thạch Sanh"], story_type: "cổ tích", origin: "Việt Nam", themes: ["anh hùng", "phép thuật", "thiện ác"], main_characters: ["Thạch Sanh", "Lý Thông", "công chúa"]|
| 3 | Hồ Gươm | https://loigiaihay.com/su-tich-ho-guom-truyen-co-tich-the-gioi-a182987.html | 7017 | story_title: ["Sự tích Hồ Gươm", "Hồ Hoàn Kiếm"], story_type: "truyền thuyết", origin: "Việt Nam", themes: ["lịch sử", "yêu nước", "thần linh"], main_characters: ["Lê Lợi", "Rùa Vàng", "Lê Thận"] |
| 4 | Ngưu Lang Chức Nữ | https://loigiaihay.com/su-tich-nguu-lang-chuc-nu-truyen-co-tich-the-gioi-a183361.html | 5284 | story_title: ["Ngưu Lang Chức Nữ"], story_type: "truyền thuyết", origin: "Trung Quốc", themes: ["tình yêu", "chia ly", "thiên đình"], main_characters: ["Ngưu Lang", "Chức Nữ", "Ngọc Hoàng"] |
| 5 | Cây Khế | https://loigiaihay.com/su-tich-cay-khe-truyen-co-tich-viet-nam-a183775.html | 10086 | story_title: ["Cây Khế"], story_type: "cổ tích", origin: "Việt Nam", themes: ["tham lam", "thiện ác", "lòng tốt"], main_characters: ["người anh", "người em", "chim phượng hoàng"] |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| story_title | list | ["Sọ Dừa"] | Lọc đúng tài liệu khi user hỏi theo tên truyện |
| story_type | string | "cổ tích" / "truyền thuyết" | Phân loại thể loại, filter theo nhóm |
| origin | string | "Việt Nam" / "Trung Quốc" | Phân biệt nguồn gốc truyện |
| main_characters | list | ["Sọ Dừa", "Phú Ông"] | Retrieve tài liệu khi user hỏi về nhân vật cụ thể |
| themes | list | ["phép thuật", "tình yêu"] | Gợi ý truyện cùng chủ đề |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu "Thạch Sanh" và "Sọ Dừa":

| Tài liệu           | Strategy                         | Chunk Count | Avg Length | Preserves Context?                                                                                                                           |
| ------------------ | -------------------------------- | ----------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| cayke.txt          | FixedSizeChunker (`fixed_size`)  | 47          | 198.2      | Cắt theo số ký tự cứng nhắc. Rất dễ cắt đôi từ hoặc cắt giữa câu, làm mất ý nghĩa.                                                           |
|                    | SentenceChunker (`by_sentences`) | 23          | 303.0      | Giữ được trọn vẹn ý nghĩa của từng câu. Tuy nhiên, mối liên hệ giữa các câu trong cùng một đoạn có thể bị mất.                               |
|                    | RecursiveChunker (`recursive`)   | 54          | 128.0      | Ưu tiên cắt theo đoạn văn (\n\n), sau đó mới đến dòng (\n) và câu. Nó giữ các thông tin có liên quan về mặt cấu trúc ở gần nhau nhất có thể. |
| nguulangchucnu.txt | FixedSizeChunker (`fixed_size`)  | 35          | 199.5      | Xuyên tạc ý nghĩa do cắt vụn giữa các đoạn hội thoại hoặc diễn biến tình cảm quan trọng của Ngưu Lang và Chúc Nữ. |
|                    | SentenceChunker (`by_sentences`) | 13          | 404.4      | Khá tốt, bảo toàn được nội dung trọn vẹn của câu nói mong nhớ, nhưng làm đứt gãy luồng cảm xúc liền mạch giữa 2 câu. |
|                    | RecursiveChunker (`recursive`)   | 41          | 127.0      | Giữ được toàn bộ diễn biến của từng phân cảnh (như cảnh chia ly ở sông Ngân) trong một khối duy nhất. |
| sodua.txt          | FixedSizeChunker (`fixed_size`)  | 38          | 196.9      | Mất ngữ cảnh về sự biến hóa về diện mạo và hành động của Sọ Dừa do chunk bị cắt cụt ở giữa dòng miêu tả. |
|                    | SentenceChunker (`by_sentences`) | 24          | 232.8      | Ổn định, nhưng làm đứt liên kết nguyên nhân - kết quả của câu chuyện. |
|                    | RecursiveChunker (`recursive`)   | 39          | 142.5      | Bao bọc toàn bộ các tình huống phép thuật kì ảo của Sọ Dừa nguyên vẹn trong một chunk. |

### Strategy Của Tôi

**Loại:** Em sử dụng sentence chunk.

**Mô tả cách hoạt động:**
> Sentence chunk này hoạt động bằng cách tách chuỗi K câu liên tiếp tạo thành một chunk. Dầu hiệu một câu sẽ là cuối câu có các dấu kết thúc câu như .!? Sentence chunk có thể kết hợp thêm overlap sentence để làm giàu thông tin cho các chunk, kết nối thông tin giữa các chunk, đảm bảo thông tin không bị bỏ sót.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Đề tài chúng em chọn là những câu chuyện cổ tích dân gian. Những câu chuyện này thường có bố cục rõ ràng, tách đoạn, tách ý câu chuyện giúp dễ dàng triển khai và đánh giá các chiến thuật chunking.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| cayke.txt  | baseline | 47 | 198.2 | 2/5 |
| cayke.txt | **của tôi** | 23 | 303.0 | 4/5 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy         | Retrieval Score (/10) | Điểm mạnh                                                                                               | Điểm yếu                                                                             |
| ---------- | ---------------- | --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Trí        | FixedSizeChunker | 4.0/10                | Tốc độ chuẩn bị dữ liệu (Index) nhanh nhất, kích cỡ chunk đều đặn dễ cấp phát bộ nhớ.                   | Đánh mất hoàn toàn bối cảnh truyện do ranh giới cắt chữ rơi ngẫu nhiên vào giữa câu. |
| Khải       | RecursiveChunker | 9.0/10                | Giữ vẹn nguyên độ liền mạch trong cốt truyện của nhân vật, bảo toàn Chunk Coherence tuyệt đối. | Kịch bản khởi tạo phức tạp và tốn CPU xử lý phép toán đệ quy.                        |
| Tuấn       | SentenceChunker  | 7.0/10                | Bắt khá chuẩn các câu "Rút Ra Bài Học" ở cuối truyện do chỉ chứa 1 dấu chấm chấm dứt.                  | Dễ gián đoạn các sự kiện có ngữ cảnh kéo dài (buộc LLM phải đọc nhiều chunk ngắt quãng).     |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker sẽ tốt nhất trong trường hợp này, vì nó giúp giữ nguyên vẹn độ liền mạch của cốt truyện. 

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em đã sử dụng biểu thức chính quy (regex) `r'(?<=[.!?])\s+'` để tách văn bản thành các câu. Regex này tìm kiếm các khoảng trắng xuất hiện ngay sau các dấu câu như `.`, `!`, hoặc `?`, giúp phân chia câu một cách chính xác. Sau khi tách, em nhóm các câu lại thành từng chunk, mỗi chunk chứa tối đa `max_sentences_per_chunk` câu để đảm bảo ngữ cảnh của từng câu được giữ trọn vẹn.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán này hoạt động theo nguyên tắc đệ quy, ưu tiên chia văn bản bằng các dấu phân tách có ý nghĩa ngữ nghĩa cao nhất trước (như `\n`). Base case của đệ quy là khi một đoạn văn bản đã nhỏ hơn `chunk_size`. Nếu không thể chia nhỏ hơn nữa bằng các dấu phân tách, nó sẽ quay về cách chia theo kích thước cố định. Cách tiếp cận này giúp giữ các đoạn văn có liên quan về mặt cấu trúc ở gần nhau nhất có thể.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi thêm tài liệu, em tính toán vector embedding cho mỗi chunk và lưu trữ chúng. Nếu có ChromaDB, em sử dụng `collection.add`. Nếu không, em lưu vào một danh sách trong bộ nhớ. Khi tìm kiếm, em tạo embedding cho câu truy vấn và tính toán độ tương đồng (tích vô hướng) với tất cả các chunk đã lưu để tìm ra các chunk phù hợp nhất.

**`search_with_filter` + `delete_document`** — approach:
> Đối với tìm kiếm có bộ lọc, em áp dụng điều kiện lọc `where` của ChromaDB để thu hẹp không gian tìm kiếm trước khi tính toán độ tương đồng. Khi xóa tài liệu, em sử dụng `collection.delete` với điều kiện `where={"doc_id": doc_id}` để xóa tất cả các chunk liên quan. Logic này đảm bảo hiệu quả bằng cách lọc trước khi thực hiện các thao tác tốn kém hơn.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc prompt của em được thiết kế để hướng dẫn LLM hoạt động như một trợ lý thông minh. Em cung cấp các chunk tìm được làm ngữ cảnh và yêu cầu LLM trả lời câu hỏi của người dùng chỉ dựa trên ngữ cảnh đó. Em cũng chỉ thị rõ ràng rằng nếu thông tin không đủ, LLM nên trả lời là không biết thay vì tự bịa ra câu trả lời, nhằm tăng tính xác thực.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sọ Dừa không có tay chân, thân hình tròn như quả dừa. | Chàng Sọ Dừa là một người kỳ lạ, không có hình dạng người thường. | high | 0.41 | ✓ |
| 2 | Thạch Sanh giương cung bắn trúng đại bàng, cứu được công chúa. | Người anh trong Cây Khế tham lam, muốn lấy hết vàng bạc châu báu. | low | 0.15 | ✓ |
| 3 | Vua Lê Lợi trả lại gươm báu cho Rùa Vàng ở hồ Tả Vọng. | Hồ Gươm là nơi diễn ra sự kiện trả gươm thần cho Long Vương. | high | 0.38 | ✓ |
| 4 | Chức Nữ là một nàng tiên dệt vải trên trời. | Người em trong Cây Khế được chim phượng hoàng trả ơn bằng vàng. | low | 0.22 | ✓ |
| 5 | Lý Thông lừa Thạch Sanh đi canh miếu thờ, hòng cướp công của chàng. | Người anh trai đối xử tệ bạc với người em, chiếm hết gia tài. | high | 0.31 | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Em dự đoán cặp 5 sẽ có điểm tương đồng cao bất ngờ. Mặc dù hai câu nói về hai nhân vật và hai câu chuyện khác nhau ("Lý Thông" và "người anh"), chúng cùng mô tả một hành động có bản chất giống nhau: sự lừa lọc, đối xử tệ bạc vì lòng tham. Điều này cho thấy embeddings có khả năng nắm bắt được ý nghĩa trừu tượng và chủ đề chung, chứ không chỉ dựa vào từ khóa bề mặt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | Là một khối thịt đỏ hỏn, không tay không chân, tròn lăn lóc giống như một quả dừa. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | Lý Thông vốn là kẻ tiểu nhân tráo trở, thấy Thạch Sanh thật thà khoẻ mạnh nên lợi dụng để cướp công giết chằn tinh nhằm tiến thân. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | Gắn liền trực tiếp với cuộc chiến của vua Lê Lợi (mệnh danh Bình Định Vương) mượn Gươm của Thần Kim Quy đánh tan quân Minh. |
| 4 | Bi kịch của Ngưu Lang và Chúc Nữ bắt nguồn từ đâu? | Bắt nguồn từ sự cấm cản của Ngọc Hoàng vì ranh giới Tiên - Phàm và trách nhiệm chốn tiên giới bị bỏ bê. |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | Lòng tham vô đáy (như người anh) sẽ chuốc lấy sự hủy diệt, còn sự chia sẻ yêu thương sẽ đơm bông kết trái bền vững. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|---|---|---|---|---|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | [book/sodua.txt] Chiều hôm đó Thạch Sanh đi đốn củi về thì Lý Thông đãi rượu, rồi bảo:.. | 0.369 | 2/5 | Tôi không biết. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | [book/caykhe.txt] Lên hòn đảo, người anh không chỉ nhét đầy túi châu báu mà cò... | 0.374 | 2/5 | Trong ngữ cảnh provided, không có thông tin nào đề cập đến lý do tại sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại. Do đó, tôi không biết. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | [book/nguulangchucnu.txt] Ngưu Lang nghe theo lời bò, đi giày làm bằng da bò, mang the... | 0.365 |1/5 | Sự tích Hồ Gươm không đề cập đến vị anh hùng lịch sử cụ thể nào trong phần ngữ cảnh được cung cấp. |
| 4 | Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ đâu? | [book/caykhe.txt] Lên hòn đảo, người anh không chỉ nhét đầy túi châu báu mà cò... | 0.331 | 3/5| Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ sự can thiệp của Vương Mẫu Nương Nương, khi bà đã dùng trâm vàng vạch ra dòng sông Thiên Hà để ngăn cách hai người, khiến họ chỉ có thể nhìn nhau từ hai bờ sông và rơi nước mắt. |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | [book/nguulangchucnu.txt] Nếu muốn vượt qua nghịch cảnh, muốn có được cuộc sống hạnh p... | 0.378 | 3/5 | Bài học rõ nét nhất từ câu chuyện "Cây Khế" là "Ở hiền gặp lành". Truyện nhấn mạnh rằng nếu chúng ta sống ngay thẳng, thật thà, thì sẽ có quý nhân phù trợ và cuộc sống của chúng ta sẽ trở nên tốt đẹp hơn. Ngoài ra, câu chuyện cũng chỉ ra rằng sự chăm chỉ nỗ lực sẽ mang lại thành quả, trong khi tham lam và lười biếng sẽ dẫn đến kết cục không may mắn. |

Kết quả chunk không được tốt lý do chính đến từ model embedding mock.
**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Qua phần so sánh, em nhận thấy chiến lược `RecursiveChunker` của bạn Khải tuy phức tạp hơn nhưng lại cực kỳ hiệu quả trong việc giữ gìn sự liền mạch của cốt truyện. Cách tiếp cận này giúp các chunk chứa đựng ngữ cảnh trọn vẹn hơn, từ đó cải thiện đáng kể chất lượng retrieval, đây là một bài học quý giá so với `SentenceChunker` mà em đã chọn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Trong buổi demo, em rất ấn tượng với một nhóm đã trình bày về việc embed thông tin chính sách của VinUni và đặc biệt là cách chunking của nhóm bạn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu có cơ hội làm lại, em sẽ đầu tư nhiều hơn vào việc làm giàu metadata ở cấp độ chunk thay vì chỉ ở cấp độ tài liệu. Ví dụ, mỗi chunk có thể được gán thêm metadata về các nhân vật xuất hiện trong đó hoặc các sự kiện chính. Điều này sẽ cho phép thực hiện các truy vấn lọc mạnh mẽ hơn, giúp hệ thống tìm kiếm chính xác hơn nữa.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
