# I. Transformer và Mô hình Sinh (Generative Models)
## A. Transformer
### 1. Định nghĩa
- Transformer là một lớp các mô hình học sâu được xác định bởi một số đặc điểm kiến trúc nhất định, lần đầu tiên được giới thiệu trong bài báo "Attention Is All You Need" bởi các nhà nghiên cứu của Google vào năm 2017. Bài báo này, với số lượng trích dẫn đáng kể , đã đánh dấu một sự thay đổi mô hình trong việc mô hình hóa chuỗi. Transformer ban đầu là một mô hình mã hóa-giải mã (encoder-decoder) , nhưng các mô hình có ảnh hưởng sau này như BERT và GPT chỉ sử dụng phần mã hóa hoặc phần giải mã tương ứng, làm nổi bật tính mô-đun và sức mạnh của kiến trúc lớp cốt lõi. Nền tảng của Transformer là cơ chế chú ý đa đầu (multi-head attention).
### 2. Kiến Trúc Transformer: Encoder, Decorder và Cơ Chế Chú Ý (Tự Chú Ý, Chú Ý Đa Đầu)
- Kiến trúc Transformer ban đầu bao gồm hai phần chính: bộ mã hóa (encoder) và bộ giải mã (decoder).
  - Bộ mã hóa (Encoder): Nhận một chuỗi các token làm đầu vào và tạo ra một biểu diễn có chiều cố định cho mỗi token, cùng với một embedding cho toàn bộ chuỗi. Nó đặt mỗi token vào ngữ cảnh trong chuỗi đầu vào.
  - Bộ giải mã (Decoder): Nhận đầu ra của bộ mã hóa và tạo ra một chuỗi các token ở đầu ra.
  - Kiến trúc Lớp Cốt Lõi: Khối xây dựng cơ bản của cả bộ mã hóa và bộ giải mã bao gồm một cơ chế tự chú ý (self-attention) và một lớp truyền thẳng (feed-forward layer). Mỗi token đầu vào đi qua các lớp này một cách độc lập nhưng lại phụ thuộc trực tiếp vào mọi token khác trong chuỗi đầu vào.
  - Cơ chế Chú ý (Attention Mechanism): Đây là sự đổi mới quan trọng, cho phép mô hình cân nhắc tầm quan trọng của các phần khác nhau trong chuỗi đầu vào khi xử lý một token cụ thể. Bài báo "Attention Is All You Need" đề xuất rằng sự chú ý có thể là cơ chế duy nhất để suy ra các phụ thuộc, loại bỏ sự cần thiết của các kết nối lặp lại. Nó tập trung có chọn lọc vào các phần liên quan của đầu vào.
  - Chú ý Đa Đầu (Multi-Head Attention): Một cải tiến nơi nhiều cơ chế chú ý ("đầu") hoạt động song song, mỗi đầu tập trung vào các khía cạnh khác nhau của đầu vào. Điều này cho phép mô hình cùng lúc chú ý đến thông tin từ các không gian con biểu diễn khác nhau tại các vị trí khác nhau. Đầu ra của các đầu này sau đó được ghép lại và biến đổi tuyến tính. Cấu trúc này tăng cường khả năng song song hóa. Ở mỗi lớp, các token được đặt trong ngữ cảnh thông qua cơ chế chú ý đa đầu song song này, khuếch đại các token quan trọng và làm giảm các token ít quan trọng hơn.
  - Chuẩn hóa Lớp (Layer Normalization): Một bài báo năm 2020 phát hiện ra rằng việc áp dụng chuẩn hóa lớp trước (thay vì sau) các lớp chú ý đa đầu và lớp truyền thẳng giúp ổn định quá trình huấn luyện và có thể không cần giai đoạn khởi động tốc độ học (learning rate warmup). Đây là một sự tinh chỉnh kiến trúc thực tế.
### 3. Ưu Điểm của Transformer so với Mô Hình Tuần Tự (RNN/LSTM)
- Transformer mang lại nhiều ưu điểm vượt trội so với các mô hình tuần tự truyền thống như Mạng Nơ-ron Hồi quy (RNN) và Bộ nhớ Dài-Ngắn hạn (LSTM), đặc biệt về khả năng song song hóa, xử lý các phụ thuộc tầm xa và khả năng mở rộng.
  - Song Song Hóa (Parallelization): Transformer xử lý tất cả các phần của một chuỗi đầu vào đồng thời bằng cách sử dụng cơ chế tự chú ý, không giống như RNN/LSTM xử lý tuần tự từng bước một. Điều này làm cho Transformer nhanh hơn đáng kể trong quá trình huấn luyện và có khả năng mở rộng tốt hơn, đặc biệt đối với các tập dữ liệu lớn.
  - Xử Lý Phụ Thuộc Tầm Xa (Handling Long-Range Dependencies): Transformer vượt trội trong việc nắm bắt các phụ thuộc tầm xa nhờ cơ chế tự chú ý, tính toán trực tiếp mối quan hệ giữa bất kỳ hai token nào trong một chuỗi, bất kể khoảng cách. RNN gặp khó khăn với vấn đề này do hiện tượng tiêu biến gradient (vanishing gradient). Mặc dù LSTM được thiết kế để giảm thiểu tiêu biến gradient và tốt hơn RNN trong việc xử lý phụ thuộc tầm xa , chúng vẫn có thể gặp khó khăn với các chuỗi rất dài và tốn kém chi phí tính toán hơn Transformer trong những trường hợp như vậy. Một số biến thể Transformer như Transformer-XL còn tăng cường hơn nữa khả năng xử lý ngữ cảnh dài.
  - Khả Năng Mở Rộng và Hiệu Quả (Scalability and Efficiency): Hiệu quả kiến trúc và khả năng xử lý song song của Transformer cho phép khả năng mở rộng lớn hơn và tốc độ xử lý nhanh hơn, làm cho chúng phù hợp với các ứng dụng quy mô lớn và các mô hình phức tạp. LSTM chậm hơn khi huấn luyện và đòi hỏi nhiều tài nguyên tính toán hơn.
  - Tài Nguyên Tính Toán (Computational Resources): Mặc dù các mô hình Transformer lớn đòi hỏi sức mạnh tính toán đáng kể (đặc biệt là GPU) để huấn luyện , bản chất song song của chúng làm cho chúng hiệu quả hơn trên mỗi phép tính so với xử lý tuần tự của LSTM đối với các tập dữ liệu rất lớn.
### 4. Ứng Dụng Nổi Bật: Từ NLP (BERT, GPT, T5) đến AI Tổng Quát
- Các mô hình Transformer đã tạo ra một cuộc cách mạng trong Xử lý Ngôn ngữ Tự nhiên (NLP), cải thiện đáng kể khả năng hiểu ngữ cảnh, độ chính xác dịch thuật và khả năng sinh văn bản.
  - BERT (Bidirectional Encoder Representations from Transformers): Được phát triển bởi Google, BERT sử dụng phần mã hóa của kiến trúc Transformer. Nó hiểu ngữ cảnh bằng cách phân tích từ theo cả hai chiều (từ trái sang phải và từ phải sang trái). BERT được sử dụng trong các thuật toán Tìm kiếm của Google, cải thiện khả năng hiểu truy vấn lên đến 30% đối với các truy vấn dài (long-tail queries). Quá trình tiền huấn luyện của BERT bao gồm việc che một số từ trong đầu vào và yêu cầu mô hình dự đoán các từ bị che đó dựa trên ngữ cảnh xung quanh.
  - Các mô hình GPT (Generative Pre-trained Transformer): Được phát triển bởi OpenAI, các mô hình GPT chủ yếu sử dụng phần giải mã của kiến trúc Transformer. Chúng xuất sắc trong việc sinh văn bản, trả lời câu hỏi và tạo ra AI đàm thoại (ví dụ: ChatGPT, GPT-4). Các doanh nghiệp báo cáo mức độ tương tác của khách hàng tăng 40% khi sử dụng chatbot được hỗ trợ bởi GPT. GPT sử dụng cơ chế che nhân quả (causal masking), trong đó mô hình dự đoán token tiếp theo dựa trên các token đã có trước đó.
  - T5 (Text-to-Text Transfer Transformer): Được phát triển bởi Google, T5 coi tất cả các tác vụ NLP là các phép biến đổi từ văn bản sang văn bản (ví dụ: "dịch tiếng Anh sang tiếng Đức: [văn bản]"). Điều này làm cho T5 trở nên linh hoạt cho các tác vụ tóm tắt văn bản, dịch thuật và trả lời câu hỏi.
  - Các ứng dụng NLP khác: Dịch máy (Google Translate sử dụng Transformer ), tóm tắt văn bản , phân tích tình cảm, chatbot , viết quảng cáo bằng AI , xử lý tài liệu.

## B.  Mô hình Sinh (Generative Models) 
### 1. Định nghĩa
- Mô hình sinh là các mô hình học máy được thiết kế để tạo ra dữ liệu mới tương tự như dữ liệu huấn luyện của chúng. Chúng học các mẫu và phân phối từ dữ liệu huấn luyện và áp dụng sự hiểu biết này để tạo ra nội dung mới lạ. Đây thường là các mạng nơ-ron tiên tiến xử lý dữ liệu huấn luyện để tạo ra các đầu ra mới. Chúng hoạt động bằng cách xác định các phân phối xác suất đồng thời của các đặc trưng trong tập huấn luyện và sau đó dựa trên kiến thức đã học này để tạo ra các mẫu dữ liệu mới. Thông thường, chúng được huấn luyện bằng phương pháp học không giám sát trên dữ liệu không có nhãn.
### 2. Models Discriminative vs. Generative
- Mô Hình Sinh (Generative Models):
  - Học phân phối xác suất đồng thời p(x,y) (hoặc p(x∣y) và p(y) để suy ra p(x,y)).
  - Mô hình hóa cách dữ liệu được tạo ra và có thể tạo ra các mẫu dữ liệu mới.
  - Trả lời câu hỏi: "Dựa trên các giả định tạo sinh của tôi, danh mục nào có khả năng tạo ra tín hiệu này nhất ".
  - Về mặt toán học, chúng có thể ước tính P(Y) và P(X∣Y) và sử dụng định lý Bayes cho P(Y∣X).
- Mô Hình Phân Biệt (Discriminative Models):
  - Học trực tiếp phân phối xác suất có điều kiện p(y∣x).
  - Tập trung vào việc học ranh giới giữa các lớp để phân biệt giữa các mẫu.
  - Trả lời câu hỏi: "Mẫu này nằm ở phía nào của ranh giới quyết định?".
  - Không quan tâm đến cách dữ liệu được tạo ra.
- Trường hợp sử dụng: Mô hình phân biệt thường được sử dụng cho các tác vụ phân loại có giám sát (ví dụ: hồi quy logistic, SVM). Mô hình sinh thường được sử dụng trong học không giám sát và cho các tác vụ như tạo điểm dữ liệu mới, phát hiện bất thường hoặc khử nhiễu hình ảnh.
- Ngoại lệ (Outliers): Mô hình sinh có thể dễ bị ảnh hưởng bởi các điểm ngoại lệ hơn, trong khi mô hình phân biệt có thể ít bị ảnh hưởng hơn vì chúng tập trung vào ranh giới
### 3.  Mạng Đối Nghịch Tạo Sinh (GAN - Generative Adversarial Networks)
- GAN bao gồm hai mạng nơ-ron: một Bộ Tạo Sinh (Generator - G) và một Bộ Phân Biệt (Discriminator - D), được huấn luyện trong một quy trình đối nghịch:
  - Bộ Tạo Sinh (Generator): Nhận nhiễu ngẫu nhiên làm đầu vào và cố gắng tạo ra dữ liệu tổng hợp (ví dụ: hình ảnh) giống với dữ liệu thật. Mục tiêu của nó là đánh lừa Bộ Phân Biệt để phân loại đầu ra của nó là thật.
  - Bộ Phân Biệt (Discriminator): Hoạt động như một bộ phân loại nhị phân, cố gắng phân biệt giữa dữ liệu thật (từ tập huấn luyện) và dữ liệu giả do Bộ Tạo Sinh tạo ra. Nó nhận dữ liệu thật làm ví dụ dương và dữ liệu giả làm ví dụ âm.
  - Huấn Luyện Đối Nghịch: Hai mạng tham gia vào một trò chơi "mèo vờn chuột" hoặc trò chơi minimax. Bộ Tạo Sinh cải thiện khả năng tạo dữ liệu thực tế, trong khi Bộ Phân Biệt trở nên tốt hơn trong việc phát hiện hàng giả. Sự cạnh tranh này thúc đẩy cả hai mạng cải thiện, dẫn đến việc tạo ra dữ liệu tổng hợp chất lượng cao. Hàm mất mát của Bộ Tạo Sinh phạt nó nếu Bộ Phân Biệt phát hiện ra hàng giả của nó, trong khi hàm mất mát của Bộ Phân Biệt phạt nó vì phân loại sai thật thành giả hoặc giả thành thật.
### 4. Ứng dụng: tạo ảnh, deepfake, dữ liệu tổng hợp
- GAN có nhiều ứng dụng thực tế, tận dụng khả năng tạo ra dữ liệu mới và chân thực:
  - Tạo Ảnh Chất Lượng Cao: GAN vượt trội trong việc tạo ra các hình ảnh thực tế (khuôn mặt, phong cảnh, nghệ thuật) thường không thể phân biệt được với ảnh thật. Ví dụ bao gồm StyleGAN để thao tác chính xác.
  - Deepfake: Tạo ra các hình ảnh, video hoặc âm thanh giả mạo siêu thực của các cá nhân. Điều này gây ra những lo ngại về đạo đức liên quan đến việc lạm dụng. Các kỹ thuật bao gồm hoán đổi khuôn mặt, tái hiện khuôn mặt và đồng bộ hóa biểu cảm khuôn mặt.
  - Tạo Dữ Liệu Tổng Hợp / Tăng Cường Dữ Liệu: Tạo dữ liệu tổng hợp để huấn luyện các mô hình ML khác, đặc biệt khi dữ liệu thật khan hiếm, đắt đỏ hoặc nhạy cảm. Điều này giúp cải thiện hiệu suất mô hình bằng cách mở rộng tập dữ liệu.
  - Các Ứng Dụng Khác: Tạo video , siêu phân giải (nâng cao chất lượng hình ảnh, SRGAN) , tạo nghệ thuật, thiết kế trò chơi điện tử, hình ảnh y tế (nâng cao chất lượng quét), tạo âm thanh/lời nói (WaveGAN, Tacotron-GAN) , diễn viên/avatar ảo
### 5. Bộ Tự Mã Hóa Biến Đổi (VAE - Variational Autoencoders)
- VAE là mô hình sinh học cách tạo dữ liệu mới bằng cách học một biểu diễn xác suất liên tục của không gian ẩn.
  - Cấu trúc Mã hóa-Giải mã: Tương tự như các bộ tự mã hóa tiêu chuẩn, VAE có một bộ mã hóa nén dữ liệu đầu vào thành một không gian ẩn có chiều thấp hơn (z) và một bộ giải mã tái tạo dữ liệu đầu vào từ biểu diễn ẩn này.
  - Bộ Mã Hóa Xác Suất: Khác với các bộ tự mã hóa tiêu chuẩn ánh xạ đầu vào thành một vectơ ẩn duy nhất, bộ mã hóa của VAE (mô hình nhận dạng) ánh xạ dữ liệu đầu vào x thành các tham số của một phân phối xác suất (thường là Gaussian) trong không gian ẩn. Nó xuất ra một vectơ trung bình (μ) và một vectơ độ lệch chuẩn (σ) cho mỗi thuộc tính ẩn.
  - Lấy Mẫu Không Gian Ẩn (Thủ Thuật Tái Tham Số Hóa - Reparameterization Trick): Để tạo một vectơ ẩn z cho bộ giải mã, một mẫu được lấy từ phân phối đã học này p(z∣x) (xấp xỉ bởi q(z∣x)). Thủ thuật tái tham số hóa (z=μ+σ⋅ϵ, trong đó ϵ được lấy mẫu từ một phân phối chuẩn tiêu chuẩn) được sử dụng để cho phép gradient lan truyền qua quá trình lấy mẫu trong quá trình lan truyền ngược.
  - Bộ Giải Mã (Mô Hình Tạo Sinh): Bộ giải mã nhận một vectơ ẩn z đã được lấy mẫu và tái tạo dữ liệu đầu vào ban đầu.
  - Hàm Mất Mát: VAE được huấn luyện bằng cách tối ưu hóa một hàm mất mát thường bao gồm hai thành phần :
    - Mất Mát Tái Tạo (Reconstruction Loss): Đo lường mức độ bộ giải mã tái tạo dữ liệu đầu vào từ biểu diễn ẩn (ví dụ: Sai số Bình phương Trung bình hoặc Binary Cross-Entropy). Điều này khuyến khích P(X∣z) cao.
    - Thành Phần Chính Quy Hóa KL Divergence: Đo lường sự khác biệt giữa phân phối ẩn đã học q(z∣x) và một phân phối tiên nghiệm p(z) (thường là một phân phối chuẩn tiêu chuẩn N(0,I)). Thành phần này khuyến khích không gian ẩn trơn tru và liên tục, phù hợp cho việc tạo sinh.
### 6. Ứng Dụng
- VAE có nhiều ứng dụng đa dạng nhờ khả năng học các biểu diễn ẩn có ý nghĩa và tạo ra dữ liệu mới.
  - Tạo Ảnh: Tạo ra các hình ảnh trông giống thật, chẳng hạn như khuôn mặt người, bằng cách học phân phối thống kê của các đặc điểm từ các tập dữ liệu hình ảnh thực tế. Hữu ích cho các ứng dụng thử đồ ảo.
  - Phát Hiện Bất Thường: Xác định các mẫu bất thường trong dữ liệu bằng cách học các mẫu bình thường và phát hiện các sai lệch (ví dụ: trong dữ liệu cảm biến công nghiệp để bảo trì dự đoán).
  - Nén Dữ Liệu: VAE có thể được sử dụng để giảm chiều dữ liệu và nén dữ liệu, mặc dù không phải lúc nào cũng là trọng tâm chính của chúng.
  - Xử Lý Ngôn Ngữ Tự Nhiên: Tạo văn bản và cho các tác vụ hiểu ngôn ngữ.
  - Học Biểu Diễn: Học các biểu diễn ẩn có chiều thấp hơn, có ý nghĩa, có thể tách rời các yếu tố biến đổi cơ bản trong dữ liệu.
  - Khám Phá Thuốc: VAE có ứng dụng trong việc tạo ra các phân tử thuốc mới.

# II. Transfer Learning và Fine-tuning
## A. Transfer Learning
### 1. Định nghĩa
- Học chuyển giao (Transfer Learning) là một kỹ thuật học máy trong đó kiến thức thu được từ một tác vụ hoặc tập dữ liệu được sử dụng để cải thiện hiệu suất và khả năng khái quát hóa của mô hình trên một tác vụ hoặc tập dữ liệu liên quan nhưng khác biệt. Nó liên quan đến việc tái sử dụng một mô hình đã được phát triển cho một tác vụ cho một tác vụ thứ hai có liên quan. Học máy truyền thống giả định rằng dữ liệu huấn luyện và kiểm tra đến từ cùng một không gian đặc trưng; học chuyển giao cho phép các mô hình được huấn luyện trên một tác vụ/dữ liệu nguồn được áp dụng cho một tác vụ/dữ liệu mục tiêu mới.
- Học chuyển giao mang lại nhiều lợi ích đáng kể, đặc biệt trong các tình huống dữ liệu hạn chế hoặc khi mục tiêu là tiết kiệm chi phí.
  - Giảm Chi Phí Tính Toán: Tái sử dụng các mô hình đã được tiền huấn luyện giúp giảm thời gian huấn luyện, yêu cầu dữ liệu, đơn vị xử lý và các tài nguyên tính toán khác. Số lượng epoch cần thiết có thể ít hơn.
  - Giải Quyết Vấn Đề Tập Dữ Liệu Nhỏ: Đặc biệt hữu ích khi tập dữ liệu mục tiêu nhỏ, vì nó loại bỏ sự cần thiết phải có các tập dữ liệu lớn để đạt được độ chính xác cao. Các mô hình tiền huấn luyện cung cấp một điểm khởi đầu mạnh mẽ.
  - Tăng Tốc Phát Triển Mô Hình: Giảm thời gian phát triển mô hình lên đến 40% (nghiên cứu của MIT được trích dẫn trong ).
  - Nâng Cao Hiệu Suất Mô Hình: Có thể cải thiện độ chính xác của mô hình lên 15-20% trong các tác vụ như phân loại hình ảnh và NLP, đặc biệt khi dữ liệu huấn luyện mục tiêu hạn chế (nghiên cứu của ScienceDirect được trích dẫn trong ).
  - Giảm Nguy Cơ Overfitting: Các mô hình được tiền huấn luyện trên các tập dữ liệu lớn và đa dạng thường ít bị overfitting hơn khi được tinh chỉnh trên các tập dữ liệu nhỏ hơn, cụ thể hơn.
  - Khả Năng Tiếp Cận cho Doanh Nghiệp Vừa và Nhỏ (SME): Dân chủ hóa AI bằng cách làm cho các công nghệ AI tiên tiến có thể tiếp cận được với các doanh nghiệp nhỏ hơn thiếu nguồn lực để huấn luyện sâu rộng từ đầu.
### 2.  Các mô hình phổ biến được dùng trong Transfer Learning (BERT, ResNet, GPT, T5)
- Nhiều mô hình tiền huấn luyện đã trở thành nền tảng cho các ứng dụng học chuyển giao trong các lĩnh vực khác nhau.
- Mô Hình NLP:
  - BERT (Bidirectional Encoder Representations from Transformers): Xuất sắc cho các tác vụ hiểu ngôn ngữ, được tinh chỉnh cho phân loại, trả lời câu hỏi, v.v..24
  - Dòng GPT (Generative Pre-trained Transformer) (ví dụ: GPT-2, GPT-4): Mạnh mẽ cho các tác vụ sinh văn bản, có thể được tinh chỉnh cho nhiều tác vụ sinh và hiểu khác nhau.24 GPT-4 có khả năng đa phương thức (văn bản và hình ảnh).
  - T5 (Text-to-Text Transfer Transformer): Coi tất cả các tác vụ NLP là văn bản sang văn bản, linh hoạt cho tóm tắt, dịch thuật, v.v..
  - RoBERTa: Phiên bản BERT được tối ưu hóa.
  - ELMo, Transformer-XL, ALBERT, XLNet, PaLM: Các mô hình NLP quan trọng khác.
- Mô Hình Thị Giác Máy Tính (thường từ torchvision.models):
  - ResNet (ví dụ: ResNet18, ResNet50): Mạng phần dư sâu, rất phổ biến cho phân loại hình ảnh.
  - MobileNetV2: Mô hình nhẹ, hiệu quả cho các ứng dụng thị giác di động và nhúng.
  - EfficientNet (B0-B7): Họ các mô hình đạt độ chính xác cao với ít tham số hơn.
  - VGG: Một kiến trúc CNN nền tảng khác.
## B. Fine-turning
### 1. Định nghĩa
- Tinh chỉnh (Fine-turning) là hành động lấy một mô hình đã được tiền huấn luyện và huấn luyện thêm (retraining) trên một tập dữ liệu mới, nhỏ hơn, dành riêng cho tác vụ để điều chỉnh các tham số của nó trong khi vẫn giữ lại chuyên môn trước đó. Nó điều chỉnh trọng số của mô hình để phù hợp hơn với tác vụ mới.

### 2. Khác biệt giữa Feature Extraction vs. Fine-tuning toàn bộ mô hình
- Trích Xuất Đặc Trưng (Feature Extraction): Sử dụng mô hình tiền huấn luyện như một bộ trích xuất đặc trưng cố định. Trọng số của các lớp tiền huấn luyện được đóng băng (không được cập nhật trong quá trình huấn luyện). Chỉ các lớp mới (thường là một đầu phân loại) được thêm vào phía trên mới được huấn luyện từ đầu bằng cách sử dụng các đặc trưng được trích xuất bởi mô hình đã đóng băng.
  - Ưu điểm: Cần ít dữ liệu hơn, chi phí tính toán thấp hơn, huấn luyện nhanh hơn, giảm nguy cơ overfitting với các tập dữ liệu nhỏ.
  - Nhược điểm: Khả năng thích ứng hạn chế nếu các đặc trưng của tác vụ mới khác biệt đáng kể, hiệu suất có thể thấp hơn nếu các đặc trưng tiền huấn luyện không tối ưu cho tác vụ mới.
- Tinh Chỉnh (Toàn Bộ hoặc Một Phần - Fine-tuning): Mở đóng băng một số hoặc tất cả các lớp tiền huấn luyện và huấn luyện lại chúng cùng với các lớp mới trên tập dữ liệu mới. Điều này cho phép mô hình điều chỉnh các đặc trưng đã học để phù hợp hơn với tác vụ mới.
  - Ưu điểm: Cải thiện hiệu suất bằng cách điều chỉnh các đặc trưng, khả năng thích ứng tốt hơn với các chi tiết cụ thể của tập dữ liệu mới, linh hoạt trong việc chọn lớp nào để tinh chỉnh.
  - Nhược điểm: Cần nhiều dữ liệu hơn để tránh overfitting, chi phí tính toán cao hơn, thời gian huấn luyện lâu hơn, điều chỉnh siêu tham số phức tạp hơn.
### 3.  Chiến lược fine-tuning hiệu quả (freeze layer, learning rate thấp,…)
- Để đạt được kết quả tốt nhất khi tinh chỉnh, một số chiến lược cần được xem xét:
- Đóng Băng Lớp (Layer Freezing): Đóng băng một cách có chiến lược một số lớp nhất định (ví dụ: các lớp đầu tiên nắm bắt các đặc trưng chung) trong khi tinh chỉnh các lớp khác (ví dụ: các lớp sau, cụ thể hơn cho tác vụ).
  - Đóng băng từ dưới lên (Freeze Bottom Layers): Giữ lại khả năng biểu đạt ngôn ngữ/hình ảnh chung từ các lớp dưới, điều chỉnh các lớp trên cho các nhu cầu cụ thể của tác vụ. Việc đóng băng 25-50% các lớp transformer dưới cùng có thể mang lại hiệu suất/hiệu quả tốt.
  - Đóng băng từ trên xuống (Freeze Top Layers): Giữ các khái niệm cấp cao, tinh chỉnh các lớp dưới (ít phổ biến hơn cho tinh chỉnh tiêu chuẩn, thường dùng để thăm dò).
  - Đóng băng xen kẽ/Đóng băng theo khoảng (Alternate Freezing/Interval Freezing): Đóng băng mỗi lớp thứ n hoặc các lớp xen kẽ.
  - Lợi ích: Có thể giảm mức tiêu thụ bộ nhớ (khoảng 30-50%) và cải thiện tốc độ huấn luyện (khoảng 20-30%) trong khi vẫn duy trì hoặc cải thiện hiệu suất so với tinh chỉnh toàn bộ mô hình hoặc LoRA (đối với LLM < 3 tỷ tham số).
- Tốc Độ Học (Learning Rate): Sử dụng tốc độ học nhỏ (ví dụ: từ 2e-5 đến 5e-5 cho tinh chỉnh LLM tiêu chuẩn) để tránh những thay đổi mạnh mẽ đối với các trọng số đã được tiền huấn luyện và tránh hiện tượng quên lãng thảm khốc.
- Khởi Động/Lịch Trình Tốc Độ Học (Learning Rate Warmup/Scheduling): Tăng dần tốc độ học ở đầu quá trình huấn   luyện (warmup) và sau đó giảm dần có thể ổn định quá trình huấn luyện. StepLR là một ví dụ về lịch trình.
- Lựa Chọn Bộ Tối Ưu Hóa (Optimizer Choice): Adam hoặc AdamW là những lựa chọn phổ biến.
- Cắt Gradient (Gradient Clipping): Ngăn chặn hiện tượng bùng nổ gradient, đặc biệt quan trọng đối với các mạng sâu hơn.
- Tinh Chỉnh Hiệu Quả Tham Số (Parameter-Efficient Fine-Tuning - PEFT): Các phương pháp như LoRA (Low-Rank Adaptation), Adapters, Selective Fine-Tuning, Additive Fine-Tuning, Reparameterization. Các phương pháp này tập trung vào việc đóng băng hầu hết các tham số của mô hình và chỉ sửa đổi một tập hợp con nhỏ hoặc thêm các mô-đun nhỏ có thể huấn luyện. Điều này làm giảm đáng kể yêu cầu về tính toán và bộ nhớ để tinh chỉnh các mô hình rất lớn.