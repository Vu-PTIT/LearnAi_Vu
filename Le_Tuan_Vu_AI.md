# I.Tìm hiểu AI
## 1.Giới thiệu về AI
- **AI (Artificial Intelligence - Trí tuệ nhân tạo)** là một lĩnh vực công nghệ tiên tiến, thuộc ngành khoa học máy tính. Mục tiêu của trí tuệ nhân tạo là phát triển các hệ thống máy tính có khả năng thực hiện các nhiệm vụ thông minh như con người, bao gồm suy nghĩ logic, giải quyết vấn đề, hiểu và giao tiếp bằng ngôn ngữ, nhận diện giọng nói, và tự học hỏi, thích nghi.  
- AI có thể chia thành 2 loại chính:  
  - AI hẹp (Narrow AI): AI được thiết kế để thực hiện một nhiệm vụ cụ thể, như nhận dạng giọng nói hoặc đề xuất sản phẩm.  
   - AI tổng quát (General AI): Có khả năng thực hiện bất kỳ nhiệm vụ trí tuệ nào mà con người làm được (hiện chưa tồn tại). AI hoạt động dựa trên các thuật toán, dữ liệu và sức mạnh tính toán, thường kết hợp với các mô hình toán học như xác suất, thống kê, và tối ưu hóa. 
- Các lĩnh vực của Ai:  
AI được chia thành nhiều lĩnh vực con, bao gồm nhưng không giới hạn ở:
  - Machine Learning (Học máy): Dạy máy tính học từ dữ liệu mà không cần lập trình rõ ràng các quy tắc.  
  -Ví dụ: Phân loại ảnh (chó hay mèo) bằng Random Forest
  - Deep Learning (Học sâu): Một nhánh của Machine Learning sử dụng mạng nơ-ron nhân tạo để xử lý dữ liệu phức tạp như hình ảnh, âm thanh.  
  -Ví dụ: Nhận diện khuôn mặt bằng Convolutional Neural Networks (CNN).
  - Natural Language Processing (Xử lý ngôn ngữ tự nhiên - NLP): Giúp máy hiểu, diễn giải và tạo ra ngôn ngữ của con người.  
  -Ví dụ: Google Translate, chatbot như Grok.
  - Computer Vision (Thị giác máy tính): Cho phép máy "nhìn" và phân tích hình ảnh hoặc video.  
  -Ví dụ: Nhận diện khuôn mặt
  - Robotics (Robot học): Kết hợp AI để điều khiển robot thực hiện các tác vụ tự động.  
  -Ví dụ: Robot hút bụi Roomba, cánh tay robot trong nhà máy.
  - Expert Systems (Hệ chuyên gia): Các hệ thống mô phỏng khả năng ra quyết định của chuyên gia con người trong một lĩnh vực cụ thể.  
  -Ví dụ: Hệ thống chẩn đoán y khoa MYCIN (1970s).
  - Reinforcement Learning (Học tăng cường): Dạy máy ra quyết định tối ưu dựa trên thử nghiệm  
  -Ví dụ: AlphaGo của DeepMind đánh bại kỳ thủ cờ vây hàng đầu 
  - Knowledge Representation (Biểu diễn tri thức):Cách lưu trữ và tổ chức thông tin để máy suy luận.  
  -Ví dụ: Ontologies trong Semantic Web.    

- Ứng dụng thực tế của AI:  
AI đã và đang thay đổi nhiều ngành công nghiệp:
  - Y tế: Dự đoán đột quỵ từ dữ liệu cảm biến (wearable devices), phân tích ảnh MRI bằng Deep Learning để phát hiện khối u.
  - Giao thông: Xe tự hành Tesla sử dụng Computer Vision và Reinforcement Learning, tối ưu hóa lộ trình giao hàng bằng thuật toán AI (UPS, FedEx).
  - Tài chính: Phát hiện giao dịch gian lận bằng anomaly detection (phát hiện bất thường), dự đoán giá cổ phiếu bằng mô hình LSTM (Long Short-Term Memory).
  - Giáo dục: Hệ thống học tập thích ứng (adaptive learning) như Duolingo, trợ lý AI chấm bài tự động (GradeScope).
   - Thương mại điện tử: Hệ thống gợi ý sản phẩm của Amazon dựa trên Collaborative Filtering, chatbot hỗ trợ khách hàng 24/7 trên Shopee.
   - Giải trí: Tạo nhạc bằng AI (AIVA - Artificial Intelligence Virtual Artist), gợi ý phim trên Netflix bằng thuật toán học máy.
## 2. Dữ liệu
- **Dữ liệu** là một tổ hợp các thông tin bao gồm chữ, số, hình ảnh,… giúp con người hình dung được tổng thể của sự vật sự việc. Dữ liệu được ứng dụng nhiều trong các ngành công nghệ, kỹ thuật và khoa học. Trong AI, dữ liệu có thể được phân loại chi tiết hơn:
  - Dữ liệu có cấu trúc: Dữ liệu trong bảng (SQL databases), ví dụ: danh sách khách hàng với tên, tuổi, địa chỉ.
  - Dữ liệu bán cấu trúc: Dữ liệu có định dạng nhưng không cố định (JSON, XML), ví dụ: log hệ thống.
  - Dữ liệu không cấu trúc: Dữ liệu tự do như hình ảnh, video, văn bản (chiếm ~80% dữ liệu thế giới).
  - Dữ liệu thời gian thực: Dữ liệu được tạo liên tục (real-time), ví dụ: dữ liệu cảm biến từ xe tự hành.
 - **Dữ liệu cần thiết và quan trọng trong Ai**:  
    - Nguồn  tri thức cho mô hình:
        - AI học từ dữ liệu giống như con người học từ kinh nghiệm.  
    Ví dụ: Để nhận diện mèo, mô hình cần hàng nghìn ảnh mèo với nhãn.  

    - Chất lượng dữ liệu quyết định chất lượng mô hình:
        - Dữ liệu sai lệch (biased data) dẫn đến kết quả sai lệch.  
   Ví dụ: Nếu dữ liệu chỉ chứa ảnh mèo trắng, mô hình sẽ không nhận diện được mèo đen.
        - Dữ liệu thiếu hoặc nhiễu (noisy data) làm giảm độ chính xác.

    - Quy trình phát triển mô hình:
        - Tập huấn luyện (Training set): 70-80% dữ liệu để dạy mô hình.
        - Tập xác nhận (Validation set): 10-15% để tinh chỉnh siêu tham số.
        - Tập kiểm tra (Test set): 10-15% để đánh giá hiệu suất cuối cùng.

    - Khả năng khái quát hóa (Generalization):
        - Dữ liệu đa dạng giúp mô hình hoạt động tốt trên các tình huống chưa từng thấy.  
    Ví dụ: AI dịch ngôn ngữ cần dữ liệu từ nhiều vùng miền
## 3. Ba loại bài toán trong Machine learning
**Phân loại**
 - Supervised Learning (Học có giám sát):
    -  Cung cấp cho máy tính cả dữ liệu đầu vào và đầu ra mong muốn (nhãn). Mục tiêu là dự đoán kết quả đầu ra cho dữ liệu mới chưa được gán nhãn.
    - Chia thành 2 nhóm:
        - Classification (Phân loại): Dự đoán giá trị rời rạc  
        -Ví dụ: phân loại văn bản, nhận dạng khuôn mặt.
        - Regression (Hồi quy): Dự đoán giá trị liên tục    
        -Ví dụ: dự đoán giá nhà, dự đoán doanh số.
    - Thuật toán: Logistic Regression, Decision Trees, Neural Networks.
 - Unsupervised Learning (Học không giám sát):
    - Xử lý vấn đề khi chúng ta không biết kết quả đầu ra mong muốn và không có nhãn cho dữ liệu. Mục tiêu là khám phá cấu trúc ẩn trong dữ liệu và tìm ra một cách biểu diễn dữ liệu phù hợp.
    -  Chia thành 2 nhóm:
        - Clustering (Phân cụm): Nhóm các đối tượng tương tự lại với nhau.   
        -Ví dụ: phân loại khách hàng, phân loại tài liệu.
        - Dimensionality Reduction (Giảm chiều): Giảm số lượng biến trong dữ liệu, giữ lại các thông tin quan trọng nhất.  
        -Ví dụ: giảm chiều dữ liệu ảnh, tìm các yếu tố tiềm ẩn trong thông tin người dùng.
    - Thuật toán: K-Means, DBSCAN, Autoencoders.
 - Reinforcement Learning (Học tăng cường):
    - Một hệ thống (agent) được xây dựng để tự động học cách tương tác với môi trường của nó, thông qua quá trình áp dụng các hành động và nhận lại phản hồi từ chính môi trường. Mục tiêu là điều khiển một agent sao cho đạt được điểm thưởng tối đa từ môi trường dựa vào các hành động mà agent thực hiện.
    - Các thành phần: Agent (tác nhân), Environment (môi trường), Action (hành động), Reward (phần thưởng).
    - Thuật toán: Q-Learning, Policy Gradient, DDPG.  

**Bảng so sánh**  
|**Tiêu chí**| **Supervised Learning**| **Unsupervised Learning** | **Reinforcement Learning**|
|-|-|-|-|
| **Dữ liệu đầu vào** | Dữ liệu có nhãn (các cặp input-output được cung cấp trước) | Dữ liệu không có nhãn | Không cần dữ liệu huấn luyện ban đầu, dựa trên tương tác  |
| **Mục tiêu**| Dự đoán đầu ra chính xác dựa trên đầu vào đã học | Tìm kiếm cấu trúc ẩn, phân cụm hoặc giảm chiều dữ liệu | Tối ưu hóa tổng phần thưởng dài hạn qua thử nghiệm  |
| **Cách thức học** | Học từ ví dụ có sẵn với nhãn rõ ràng | Tự khám phá mẫu hình mà không cần hướng dẫn| Học qua thử và sai, dựa trên phản hồi từ môi trường|
| **Ví dụ ứng dụng** | - Phân loại email (spam/not spam) | - Phân cụm khách hàng theo hành vi mua sắm| - Huấn luyện AI chơi game (AlphaGo, DeepMind) |
|| - Dự báo giá nhà dựa trên diện tích, vị trí | - Phát hiện bất thường trong dữ liệu giao dịch ngân hàng | - Điều khiển robot tự hành trong nhà máy |
|| - Nhận diện chữ viết tay (OCR) | - Nén dữ liệu hình ảnh hoặc video | - Tối ưu hóa lộ trình giao hàng tự động|
| **Thuật toán phổ biến**     | - Linear Regression (Hồi quy tuyến tính)| - K-Means Clustering (Phân cụm K-Means)| - Q-Learning|
|| - Logistic Regression (Hồi quy logistic) | - Hierarchical Clustering (Phân cụm phân cấp)| - SARSA (State-Action-Reward-State-Action) |
|| - Support Vector Machines (SVM)| - Principal Component Analysis (PCA) | - Deep Q-Networks (DQN)|
|| - Decision Trees (Cây quyết định)| - Autoencoders (Mã hóa tự động)  | - Policy Gradient Methods |
|| - Neural Networks (Mạng nơ-ron) | - DBSCAN (Phân cụm dựa trên mật độ)| - Actor-Critic Methods |
| **Độ phức tạp tính toán** | Trung bình (phụ thuộc vào kích thước dữ liệu có nhãn) | Trung bình (phụ thuộc vào cấu trúc dữ liệu) | Cao (yêu cầu mô phỏng hoặc tương tác môi trường)|
| **Yêu cầu dữ liệu** | Dữ liệu chất lượng cao, được gắn nhãn chính xác  | Dữ liệu lớn nhưng không cần nhãn | Không cần dữ liệu tĩnh, cần môi trường tương tác|
| **Ưu điểm** | - Kết quả dễ đo lường và đánh giá| - Khám phá dữ liệu mới mà không cần chuẩn bị nhãn | - Linh hoạt trong môi trường động, không cần dữ liệu sẵn|
|| - Hiệu quả khi có dữ liệu huấn luyện tốt| - Hữu ích khi dữ liệu không có nhãn sẵn | - Có thể giải quyết bài toán phức tạp, dài hạn |
| **Nhược điểm**| - Cần nhiều thời gian và tài nguyên để gắn nhãn dữ liệu | - Kết quả khó đánh giá khách quan| - Tốn kém tài nguyên tính toán và thời gian|
|| - Không hiệu quả nếu dữ liệu nhãn kém chất lượng | - Dễ bị ảnh hưởng bởi nhiễu (noise) trong dữ liệu| - Khó điều chỉnh tham số và ổn định mô hình |
## 4. Các thuật ngữ (Terminology) và ký hiệu (Notions) trong AI
- Thuật ngữ (Terminology) :
    - Artificial Intelligence (AI) - Trí tuệ nhân tạo: Khả năng của máy móc mô phỏng trí thông minh của con người, như học tập, suy luận, và ra quyết định.
    - Machine Learning (ML) - Học máy: Một nhánh của AI, cho phép hệ thống học hỏi và cải thiện từ dữ liệu mà không cần lập trình chi tiết.
    - Deep Learning (DL) - Học sâu: Một tập hợp con của ML, sử dụng mạng nơ-ron nhân tạo với nhiều lớp để xử lý dữ liệu phức tạp.
    - Neural Network (NN) - Mạng nơ-ron: Hệ thống mô phỏng cách bộ não con người xử lý thông tin, bao gồm các nút (neuron) kết nối với nhau.
    - Supervised Learning - Học có giám sát: Phương pháp học máy sử dụng dữ liệu đã được gắn nhãn để huấn luyện mô hình.
    - Unsupervised Learning - Học không giám sát: Phương pháp học máy tìm kiếm cấu trúc ẩn trong dữ liệu không được gắn nhãn.
    - Reinforcement Learning (RL) - Học tăng cường: Phương pháp học máy nơi tác nhân học cách đưa ra quyết định bằng cách thử và sai trong môi trường để tối đa hóa phần thưởng.
    - Natural Language Processing (NLP) - Xử lý ngôn ngữ tự nhiên: Lĩnh vực AI tập trung vào việc máy móc hiểu và tạo ra ngôn ngữ của con người.
    - Computer Vision - Thị giác máy tính: Lĩnh vực AI cho phép máy móc "nhìn" và diễn giải hình ảnh hoặc video.
    - Algorithm - Thuật toán: Tập hợp các bước hoặc quy tắc để giải quyết một vấn đề hoặc thực hiện một nhiệm vụ.
    - Training Data - Dữ liệu huấn luyện: Tập dữ liệu được sử dụng để dạy mô hình AI học các mẫu hoặc quy luật.
    - Test Data - Dữ liệu kiểm tra: Tập dữ liệu dùng để đánh giá hiệu suất của mô hình AI sau khi huấn luyện.
    - Overfitting - Quá khớp: Hiện tượng mô hình học quá tốt trên dữ liệu huấn luyện nhưng hoạt động kém trên dữ liệu mới.
    - Underfitting - Dưới khớp: Hiện tượng mô hình không học đủ từ dữ liệu huấn luyện, dẫn đến hiệu suất kém.
    - Feature - Đặc trưng: Thuộc tính hoặc biến số trong dữ liệu được sử dụng để huấn luyện mô hình.
    - Label - Nhãn: Giá trị hoặc danh mục được gán cho dữ liệu trong học có giám sát.
    - Gradient Descent - Hạ gradient: Phương pháp tối ưu hóa để điều chỉnh tham số mô hình nhằm giảm thiểu sai số.
    - Epoch - Epoch: Một lần lặp qua toàn bộ dữ liệu huấn luyện trong quá trình học.
    - Loss Function - Hàm mất mát: Đo lường mức độ sai lệch giữa dự đoán của mô hình và giá trị thực tế.
    - Activation Function - Hàm kích hoạt: Quyết định đầu ra của một nơ-ron trong mạng nơ-ron (ví dụ: sigmoid, ReLU).
    - Convolutional Neural Network (CNN) - Mạng nơ-ron tích chập: Loại mạng nơ-ron chuyên xử lý hình ảnh và dữ liệu không gian.
    - Recurrent Neural Network (RNN) - Mạng nơ-ron hồi quy: Loại mạng nơ-ron phù hợp với dữ liệu tuần tự như chuỗi thời gian hoặc văn bản.
    - Generative AI - AI tạo sinh: Hệ thống AI tạo ra nội dung mới như văn bản, hình ảnh, âm thanh (ví dụ: GPT, DALL-E).
    - GAN (Generative Adversarial Network) - Mạng đối kháng tạo sinh: Mô hình gồm hai mạng (tạo sinh và phân biệt) cạnh tranh để tạo dữ liệu giống thật.
    - Bias - Độ chệch: Sai lệch trong mô hình do giả định không chính xác hoặc dữ liệu không cân bằng.
    - Variance - Phương sai: Mức độ nhạy cảm của mô hình với những thay đổi nhỏ trong dữ liệu huấn luyện.
    - Hyperparameter - Siêu tham số: Các tham số được thiết lập trước khi huấn luyện mô hình (ví dụ: tốc độ học).
    - Transfer Learning - Học chuyển giao: Sử dụng mô hình đã được huấn luyện trước cho một nhiệm vụ mới.
    - Embedding - Nhúng: Biểu diễn dữ liệu (như từ hoặc hình ảnh) dưới dạng vector trong không gian số chiều thấp.
    - Tokenization - Token hóa: Quá trình chia nhỏ văn bản thành các đơn vị nhỏ hơn (token) để xử lý trong NLP.
    - Backpropagation - Lan truyền ngược: Phương pháp điều chỉnh trọng số trong mạng nơ-ron dựa trên sai số đầu ra.
    - Inference - Suy luận: Quá trình sử dụng mô hình đã huấn luyện để đưa ra dự đoán trên dữ liệu mới.
    - Regularization - Điều chuẩn: Kỹ thuật giảm thiểu quá khớp (ví dụ: L1, L2 regularization).
    - Dataset - Tập dữ liệu: Bộ sưu tập dữ liệu dùng để huấn luyện, kiểm tra hoặc đánh giá mô hình AI.
    - Big Data - Dữ liệu lớn: Tập hợp dữ liệu khổng lồ được sử dụng để huấn luyện các hệ thống AI phức tạp.
    - Chatbot - Trợ lý trò chuyện: Ứng dụng AI mô phỏng cuộc hội thoại với con người.
    - Autonomous System - Hệ thống tự trị: Hệ thống AI có khả năng hoạt động độc lập mà không cần sự can thiệp của con người.
    - Ethics in AI - Đạo đức trong AI: Các nguyên tắc và vấn đề liên quan đến việc sử dụng AI một cách công bằng và an toàn.
    - ERT: Mô hình ngôn ngữ tiền huấn luyện phổ biến
    - Object Detection: Phát hiện và định vị đối tượng trong ảnh
    - Semantic Segmentation: Phân đoạn ảnh theo ngữ nghĩa
    - YOLO: Kiến trúc phát hiện đối tượng real-time
    - Preprocessing: Quá trình làm sạch và chuẩn hóa dữ liệu trước khi huấn luyện.
    - Data Augmentation: Kỹ thuật tạo thêm dữ liệu huấn luyện bằng cách biến đổi dữ liệu gốc.
    - Bias: Độ chệch trong dữ liệu hoặc mô hình do giả định sai lệch.
    - Variance: Mức độ thay đổi của mô hình khi dữ liệu huấn luyện thay đổi.
    - Regularization: Kỹ thuật giảm thiểu quá khớp (ví dụ: L1, L2).
    - Inference: Quá trình sử dụng mô hình đã huấn luyện để dự đoán trên dữ liệu mới.
    - Precision: Tỷ lệ dự đoán đúng trong số các dự đoán dương tính.
    - Recall: Tỷ lệ dự đoán đúng dương tính trên tổng số mẫu thực sự dương tính.
    - F1 Score: Trung bình hài hòa giữa Precision và Recall.
    - Dropout: Kỹ thuật ngắt ngẫu nhiên một số neuron trong quá trình huấn luyện để tránh quá khớp.
    - Pooling: Kỹ thuật giảm kích thước dữ liệu trong CNN (ví dụ: Max Pooling).
    - Attention Mechanism: Cơ chế tập trung vào các phần quan trọng của dữ liệu đầu vào.
    - Optimizer: Thuật toán tối ưu hóa như SGD, Adam, RMSprop.
    - Momentum: Kỹ thuật tăng tốc Gradient Descent bằng cách xem xét hướng di chuyển trước đó.
    - Convergence: Trạng thái mô hình đạt được khi sai số không còn giảm đáng kể
    - Word2Vec: Mô hình nhúng từ dựa trên ngữ cảnh.
    - Seq2Seq: Kiến trúc chuyển đổi chuỗi này sang chuỗi khác (ví dụ: dịch máy).
    - Language Model: Mô hình dự đoán xác suất của chuỗi từ.
    - Image Classification: Phân loại nội dung của hình ảnh.
    - Feature Extraction: Trích xuất các đặc trưng quan trọng từ hình ảnh.
    - OCR (Optical Character Recognition): Nhận diện văn bản trong hình ảnh.
    - Q-Learning: Thuật toán học tăng cường dựa trên giá trị hành động.
    - Exploration: Quá trình thử nghiệm các hành động mới để học.
    - Exploitation: Sử dụng kiến thức hiện có để tối ưu phần thưởng.
    - State: Trạng thái hiện tại của môi trường mà agent quan sát.

- Ký hiệu (Notion)
    - x: Biến đầu vào (input), thường là một vector hoặc ma trận dữ liệu.
    - y: Biến đầu ra (output), giá trị mục tiêu hoặc nhãn (label).
    - ŷ: Dự đoán đầu ra của mô hình (predicted output).
    - w: Trọng số (weight) trong mô hình, thường là vector hoặc ma trận.
    - b: Độ chệch (bias), giá trị bổ sung trong mô hình tuyến tính hoặc mạng nơ-ron.
    - $ \theta $ : Tập hợp các tham số của mô hình (thường bao gồm cả w và b).
    - n: Số lượng mẫu trong tập dữ liệu (number of samples).
    - m: Số lượng đặc trưng (features) trong mỗi mẫu.
    - X: Ma trận dữ liệu đầu vào, thường có kích thước n x m (n mẫu, m đặc trưng).
    - Y: Vector hoặc ma trận nhãn đầu ra, kích thước n x 1 hoặc n x k (k lớp trong phân loại)
    - D: Tập dữ liệu (dataset), thường được chia thành D<sub>train</sub>, D<sub>test</sub>,  D<sub>var</sub> (huấn luyện, kiểm tra, xác nhận).
    - f(x): Hàm mô hình (model function), ánh xạ từ đầu vào đến đầu ra.
    - L: Hàm mất mát (loss function), đo lường sai số giữa y va y'

# II.Tìm hiểu về một số thư viện phổ biến
## 1.NumPy
- Khái niệm:
    - NumPy là thư viện cốt lõi trong hệ sinh thái khoa học của Python, cung cấp đối tượng mảng đa chiều (ndarray) và các hàm toán học để thao tác trên mảng. Nó được viết chủ yếu bằng C, giúp tăng hiệu suất tính toán so với Python thuần.
- Thời gian ra đời và người sáng lập: 
    - NumPy được Travis Oliphant phát hành chính thức vào năm 2006. Nó là sự kết hợp giữa Numeric (do Jim Hugunin phát triển năm 1995) và Numarray, nhằm tạo ra một thư viện thống nhất cho tính toán số học trong Python.
- Ưu điểm:
    - Hiệu suất vượt trội: Sử dụng mảng ndarray được viết bằng C, tối ưu hóa tốc độ tính toán so với danh sách Python thông thường (nhanh hơn gấp 50 lần trong một số phép toán).
    - Broadcasting: Cho phép thực hiện phép toán trên mảng có kích thước khác nhau mà không cần vòng lặp thủ công, tiết kiệm thời gian và mã nguồn.
    - Tích hợp sâu với hệ sinh thái: Là nền tảng cho hầu hết các thư viện khoa học như SciPy, Scikit-learn, và thậm chí PyTorch/TensorFlow (với tensor dựa trên NumPy).
    - Hỗ trợ toán học nâng cao: Bao gồm các hàm đại số tuyến tính (như nhân ma trận, nghịch đảo), biến đổi Fourier, và số ngẫu nhiên.
    - Đa nền tảng và nhẹ: Dung lượng nhỏ, dễ cài đặt, hoạt động tốt trên cả hệ thống yếu. 
- Nhược điểm: 
    - Không hỗ trợ dữ liệu không đồng nhất: Chỉ hoạt động tốt với dữ liệu số (numeric), không phù hợp cho dữ liệu dạng chuỗi hoặc hỗn hợp (như Pandas).
    - Thiếu tính năng cấp cao: Không có công cụ xử lý dữ liệu dạng bảng, trực quan hóa, hay nhập/xuất dữ liệu từ file.
    - Khó khăn với người mới: Yêu cầu hiểu về lập trình mảng và chỉ số (indexing), có thể phức tạp khi xử lý mảng nhiều chiều.
    - Không tối ưu cho dữ liệu lớn phân tán: Không hỗ trợ tính toán song song hoặc phân tán như Dask hay Spark.
## 2.Pandas
- Khái niệm: 
    - Pandas là thư viện phân tích dữ liệu mạnh mẽ, cung cấp hai cấu trúc chính: Series (chuỗi 1 chiều) và DataFrame (bảng 2 chiều). Nó được thiết kế để xử lý dữ liệu thực tế, thường không đồng nhất hoặc thiếu sót.
- Thời gian ra đời và người sáng lập: 
    - Pandas được Wes McKinney phát hành vào năm 2008 khi anh làm việc tại AQR Capital Management, với mục tiêu tạo ra công cụ phân tích dữ liệu tài chính hiệu quả.
- Ưu điểm:
    - Xử lý dữ liệu linh hoạt: DataFrame và Series cho phép thao tác dữ liệu giống bảng tính Excel, dễ dàng lọc, nhóm, và trộn dữ liệu.
    - Quản lý dữ liệu thiếu: Các hàm như fillna(), dropna(), và interpolate() giúp xử lý dữ liệu không đầy đủ một cách hiệu quả.
    - Tích hợp I/O mạnh mẽ: Hỗ trợ đọc/ghi nhiều định dạng (CSV, Excel, JSON, SQL, HDF5) với cú pháp đơn giản, ví dụ: pd.read_csv().
    - Tối ưu cho phân tích nhanh: Cung cấp các hàm thống kê tích hợp như mean(), median(), describe() để phân tích dữ liệu tức thì.
    - Hỗ trợ thời gian: Tích hợp tốt với chuỗi thời gian (time series), hữu ích trong tài chính và IoT.
- Nhược điểm:
    - Hiệu suất giảm với dữ liệu lớn: Khi làm việc với hàng triệu dòng, Pandas chậm hơn đáng kể so với NumPy hoặc các công cụ như Dask do phụ thuộc vào RAM.
    - Không hỗ trợ GPU: Không tận dụng được sức mạnh tính toán của GPU, giới hạn trong các tác vụ nặng.
    - Tốn bộ nhớ: DataFrame lưu trữ dữ liệu không đồng nhất nên tiêu tốn nhiều RAM hơn mảng NumPy.
    - Khó debug với dữ liệu phức tạp: Khi xử lý dữ liệu lớn với nhiều phép biến đổi, việc tìm lỗi có thể phức tạp.
    - Hạn chế tính toán song song: Không tối ưu cho xử lý đa luồng hoặc phân tán.
## 3.Matplotlib
- Khái niệm: 
    - Matplotlib là thư viện trực quan hóa dữ liệu 2D (và một phần 3D) trong Python, lấy cảm hứng từ MATLAB. Nó cho phép người dùng tạo biểu đồ chuyên nghiệp để phân tích và trình bày dữ liệu.
- Thời gian ra đời và người sáng lập: 
    - Matplotlib được John D. Hunter phát hành lần đầu vào năm 2003. Hunter, một nhà khoa học thần kinh, muốn tạo công cụ trực quan hóa dữ liệu mã nguồn mở thay thế MATLAB.
- Ưu điểm:
    - Tùy chỉnh chi tiết: Cho phép điều chỉnh từng yếu tố của biểu đồ (trục, màu sắc, chú thích, kích thước) thông qua API cấp thấp.
    - Hỗ trợ đa dạng biểu đồ: Từ biểu đồ cơ bản (line, bar) đến nâng cao (contour, 3D plots), đáp ứng hầu hết nhu cầu trực quan hóa.
    - Tích hợp tốt với hệ sinh thái Python: Làm việc mượt mà với NumPy, Pandas, và Jupyter Notebook (hiển thị biểu đồ inline).
    - Xuất file linh hoạt: Có thể lưu biểu đồ dưới nhiều định dạng (PNG, JPG, PDF, SVG) với chất lượng cao, phù hợp cho xuất bản.
    - Miễn phí và mã nguồn mở: Không cần chi phí bản quyền, cộng đồng lớn hỗ trợ phát triển.
- Nhược điểm:
    - Cú pháp dài dòng: Tạo biểu đồ phức tạp đòi hỏi nhiều dòng mã, ví dụ: điều chỉnh trục hoặc thêm chú thích.
    - Giao diện mặc định kém hấp dẫn: Màu sắc và kiểu dáng cơ bản không hiện đại, cần thư viện bổ trợ như Seaborn để cải thiện.
    - Hạn chế tương tác: Không hỗ trợ tốt các biểu đồ động hoặc tương tác (so với Plotly, Bokeh), chỉ phù hợp cho biểu đồ tĩnh.
    - Hiệu suất chậm với dữ liệu lớn: Khi vẽ hàng chục nghìn điểm dữ liệu, tốc độ giảm rõ rệt.
    - Khúc học tập dốc: Người mới có thể gặp khó khăn khi làm quen với các khái niệm như Figure, Axes.
## 4.PyTorch
- Khái niệm: 
    - PyTorch là thư viện học sâu mã nguồn mở, nổi bật với tính năng “tính toán động” (dynamic computation graph), cho phép thay đổi cấu trúc mô hình trong quá trình chạy. Nó cạnh tranh trực tiếp với TensorFlow.
- Thời gian ra đời và người sáng lập: 
    - PyTorch ra mắt vào năm 2016, được phát triển bởi nhóm Facebook AI Research (FAIR), với các tác giả chính là Adam Paszke, Sam Gross, và Soumith Chintala. Nó dựa trên thư viện Torch (dùng Lua).
- Ưu điểm:
    - Tính toán động: Dynamic computation graph (eager execution) cho phép thay đổi mô hình trong lúc chạy, rất hữu ích trong nghiên cứu và thử nghiệm.
    - Dễ debug: Tích hợp tốt với Python, cho phép sử dụng các công cụ như pdb hoặc in giá trị tensor trực tiếp, không cần biên dịch trước như TensorFlow 1.x.
    - Hỗ trợ GPU mạnh mẽ: Tận dụng CUDA để tăng tốc huấn luyện mạng nơ-ron, phù hợp với các mô hình lớn.
    - API thân thiện: Gần với lập trình Python truyền thống, dễ học hơn so với các framework như Theano.
    - Cộng đồng nghiên cứu lớn: Được ưa chuộng trong học thuật, với nhiều bài báo và mã nguồn mẫu sử dụng PyTorch.
    - Tích hợp tốt với hệ sinh thái: Hỗ trợ NumPy, torchvision (cho thị giác máy tính), và torchtext (cho NLP).
- Nhược điểm:
    - Triển khai sản phẩm hạn chế: TorchServe chưa mạnh bằng TensorFlow Serving hoặc ONNX trong việc đưa mô hình vào sản xuất.
    - Tiêu tốn tài nguyên: Do tính toán động, PyTorch có thể chậm hơn TensorFlow trong một số trường hợp tối ưu hóa tĩnh.
    - Thiếu công cụ quản lý lớn: Không có hệ sinh thái đầy đủ như TensorFlow (TensorBoard, TFX) để giám sát và quản lý quy trình học máy.
    - Tài liệu chưa hoàn thiện ban đầu: Dù đã cải thiện, tài liệu vẫn không chi tiết bằng TensorFlow ở một số khía cạnh nâng cao.
    - Khó tối ưu cho thiết bị di động: Việc triển khai mô hình PyTorch trên thiết bị nhúng hoặc di động phức tạp hơn so với TensorFlow Lite.
# III.Tìm hiểu về Notebook và nguồn dữ liệu
## 1.Làm quen với Notebook
## 2.Nguồn dữ liệu (Data Sources)
a. Kaggle
- Kaggle
    - là một nền tảng trực tuyến nổi tiếng dành cho các nhà khoa học dữ liệu, kỹ sư học máy và những người đam mê trí tuệ nhân tạo (AI). Được thành lập vào năm 2010 và sau đó được Google mua lại vào năm 2017, Kaggle cung cấp một môi trường để người dùng tìm kiếm, chia sẻ và sử dụng các tập dữ liệu (dataset), tham gia các cuộc thi học máy, học hỏi qua các khóa học ngắn, và cộng tác với cộng đồng toàn cầu. Đây là một trong những cộng đồng khoa học dữ liệu lớn nhất thế giới, với hơn 5 triệu thành viên (tính đến thời điểm gần đây).
    - Kaggle không chỉ là nơi cung cấp dữ liệu mà còn là một sân chơi để kiểm tra kỹ năng phân tích dữ liệu và học máy thông qua các cuộc thi (competitions) với giải thưởng hấp dẫn. Ngoài ra, Kaggle hỗ trợ môi trường lập trình trực tuyến qua Kaggle Notebooks (dựa trên Jupyter Notebook), tích hợp GPU miễn phí để huấn luyện mô hình
- Các tính năng chính của Kaggle:
    - Datasets: Kho dữ liệu miễn phí và đa dạng, người dùng có thể tải xuống hoặc tải lên dataset của riêng mình.
    - Competitions: Các cuộc thi giải quyết vấn đề thực tế với phần thưởng tiền mặt hoặc danh tiếng.
    - Notebooks: Công cụ lập trình trực tuyến để phân tích dữ liệu và xây dựng mô hình.
    - Courses: Các khóa học miễn phí về khoa học dữ liệu, Python, SQL, v.v.
    - Discussion: Diễn đàn để trao đổi ý tưởng và học hỏi từ cộng đồng.
- Một số dataset phổ biến:
    - Titanic - Machine Learning from Disaster:
        - Mô tả: Bộ dữ liệu chứa thông tin về hành khách trên tàu Titanic (tuổi, giới tính, hạng vé, v.v.) và liệu họ có sống sót hay không. Đây là dataset kinh điển cho người mới bắt đầu học phân tích dữ liệu và học máy.
        - Ứng dụng: Dự đoán khả năng sống sót (classification).
    - House Prices - Advanced Regression Techniques:
        - Mô tả: Dữ liệu về giá nhà ở Ames, Iowa, với hơn 79 biến mô tả các đặc điểm của ngôi nhà (diện tích, số phòng, năm xây dựng, v.v.).
        - Ứng dụng: Dự đoán giá nhà (regression).
    - MNIST Digit Recognizer:
        - Mô tả: Bộ dữ liệu gồm 70.000 hình ảnh chữ số viết tay (0-9), mỗi hình ảnh là ma trận 28x28 pixel.
        - Ứng dụng: Phân loại hình ảnh (image classification).
    - Credit Card Fraud Detection:
        - Mô tả: Bộ dữ liệu ẩn danh về giao dịch thẻ tín dụng, với mục tiêu phát hiện giao dịch gian lận (rất không cân bằng - imbalanced dataset).
        - Ứng dụng: Phát hiện bất thường (anomaly detection).
    - COVID-19 Open Research Dataset (CORD-19):
        - Mô tả: Bộ sưu tập các bài báo khoa học liên quan đến COVID-19, bao gồm văn bản và siêu dữ liệu.
        - Ứng dụng: Xử lý ngôn ngữ tự nhiên (NLP) và phân tích văn bản.
b.Hugging Face
- Hugging Face
    - là một công ty và nền tảng mã nguồn mở tập trung vào xử lý ngôn ngữ tự nhiên (NLP), học máy (machine learning), và trí tuệ nhân tạo (AI). Được thành lập vào năm 2016 bởi Clément Delangue, Julien Chaumond, và Thomas Wolf, Hugging Face ban đầu là một chatbot AI, nhưng sau đó đã chuyển hướng thành một trung tâm phát triển và chia sẻ các công cụ AI, đặc biệt là thư viện Transformers. Hiện nay, nó được xem như "GitHub của AI" nhờ kho tài nguyên khổng lồ dành cho cộng đồng nghiên cứu và phát triển AI
    - Hugging Face nổi tiếng với việc cung cấp các mô hình học sâu (deep learning models) được huấn luyện sẵn (pre-trained models), bộ dữ liệu, và các công cụ mã nguồn mở giúp đơn giản hóa việc xây dựng và triển khai các ứng dụng AI, đặc biệt trong lĩnh vực NLP.
- Hugging Face cung cấp nhiều tài nguyên và công cụ quan trọng, bao gồm:
    - Thư viện Transformers:
        - Là sản phẩm nổi bật nhất của Hugging Face, một thư viện Python mã nguồn mở hỗ trợ hàng nghìn mô hình học sâu được huấn luyện sẵn (như BERT, GPT, T5, v.v.).   
        - Ứng dụng: Phân loại văn bản, dịch máy, trả lời câu hỏi, tạo văn bản, v.v.
        - Đặc điểm: Dễ sử dụng, tích hợp với PyTorch, TensorFlow, và JAX.
    - Datasets:
        - Kho lưu trữ các bộ dữ liệu công khai dành cho NLP và các tác vụ AI khác.
        - Ví dụ: SQuAD (Stanford Question Answering Dataset), MNLI (Multi-Genre Natural Language Inference), GLUE, v.v.
        - Công cụ đi kèm: Thư viện datasets để tải, xử lý và quản lý dữ liệu dễ dàng.
    - Model Hub:
        - Nơi lưu trữ hơn 100.000 mô hình AI được cộng đồng đóng góp, từ mô hình ngôn ngữ lớn (LLM) đến mô hình xử lý hình ảnh, âm thanh.
        - Người dùng có thể tải về, tinh chỉnh (fine-tune), hoặc triển khai trực tiếp qua API.
    - Spaces:
        - Một nền tảng để tạo và chia sẻ ứng dụng AI tương tác (demos) dựa trên các mô hình của Hugging Face, sử dụng Gradio hoặc Streamlit.
    - Tokenizers:
        - Công cụ xử lý văn bản nhanh và hiệu quả, hỗ trợ tokenization cho các mô hình ngôn ngữ.
    - Cộng đồng và tài liệu:
        - Diễn đàn, tài liệu chi tiết, và các khóa học giúp người dùng từ mới bắt đầu đến chuyên gia dễ dàng tiếp cận công nghệ AI.
# IV.Phân tích dữ liệu sơ bộ - Exploratory Data Analysis (EDA)
## 1.Hiểu dữ liệu (Data Understanding)
- là bước đầu tiên và quan trọng nhất trong EDA. Đây là giai đoạn bạn làm quen với tập dữ liệu, nắm bắt cấu trúc, nội dung và ý nghĩa của nó trước khi đi sâu vào phân tích. Mục tiêu là trả lời câu hỏi: "Dữ liệu của tôi trông như thế nào?"
    - Thu thập thông tin cơ bản: Bạn bắt đầu bằng cách xem xét nguồn gốc dữ liệu (dữ liệu từ đâu, ai thu thập, mục đích ban đầu là gì), kích thước (số lượng hàng, cột), và định dạng (CSV, JSON, cơ sở dữ liệu, v.v.).
    - Kiểm tra cấu trúc: Xác định các biến (cột) trong dữ liệu, bao gồm loại dữ liệu của chúng (số, văn bản, ngày tháng, phân loại, v.v.). Ví dụ, một cột "tuổi" là số nguyên, còn "giới tính" là phân loại (categorical).
    - Xác định giá trị thiếu (missing values): Kiểm tra xem có dữ liệu nào bị thiếu không (NaN, null) và đánh giá mức độ ảnh hưởng. Ví dụ, nếu 30% giá trị trong cột "thu nhập" bị thiếu, điều này có thể ảnh hưởng lớn đến phân tích sau này.
    - Nhận diện giá trị bất thường (outliers): Tìm các giá trị không hợp lý hoặc khác biệt lớn so với phần còn lại, như một người có "tuổi" là 200 trong tập dữ liệu về dân số.
    - Hiểu ngữ cảnh: Liên kết dữ liệu với thực tế. Ví dụ, nếu bạn phân tích dữ liệu bán hàng, bạn cần biết các yếu tố như mùa vụ, chương trình khuyến mãi, hoặc xu hướng thị trường có thể ảnh hưởng đến số liệu.
- Ví dụ thực tế: Giả sử bạn có tập dữ liệu về khách hàng của một cửa hàng. Bạn sẽ kiểm tra các cột như "ID khách hàng", "số tiền chi tiêu", "ngày mua hàng", và xem liệu có giá trị nào bất thường (như số tiền âm) hay không.
## 2.Phân tích dữ liệu (Data Analysis)
- Sau khi hiểu dữ liệu, bạn bắt đầu phân tích để khám phá các đặc điểm, xu hướng và mối quan hệ trong dữ liệu. Đây là giai đoạn "đào sâu" để tìm ra những gì dữ liệu đang cố gắng "nói".
    - Thống kê mô tả (Descriptive Statistics): Tính toán các giá trị cơ bản như trung bình, trung vị, độ lệch chuẩn, giá trị nhỏ nhất/lớn nhất để hiểu phân bố dữ liệu. Ví dụ, trung bình chi tiêu của khách hàng là 500.000 VNĐ, nhưng độ lệch chuẩn lớn cho thấy sự biến động mạnh.
    - Phân tích phân bố (Distribution Analysis): Xem dữ liệu phân bố như thế nào (chuẩn, lệch trái, lệch phải). Điều này giúp phát hiện xu hướng tự nhiên, như đa số khách hàng chi tiêu dưới 200.000 VNĐ nhưng có một số ít chi tiêu hàng triệu.
    - Tìm mối quan hệ (Correlation Analysis): Kiểm tra xem các biến có liên quan đến nhau không. Ví dụ, "tuổi" và "số tiền chi tiêu" có tương quan dương (người lớn tuổi chi tiêu nhiều hơn) hay không? Công cụ phổ biến là ma trận tương quan (correlation matrix).
    - Phân đoạn dữ liệu (Segmentation): Chia dữ liệu thành các nhóm để phân tích chi tiết hơn, như phân tích theo giới tính, khu vực, hoặc nhóm tuổi.
    - Kiểm tra giả thuyết sơ bộ: Đặt câu hỏi và kiểm tra nhanh, ví dụ: "Liệu khách hàng nam có chi tiêu nhiều hơn nữ không?" bằng cách so sánh trung bình giữa hai nhóm.
- Ví dụ thực tế: Trong dữ liệu bán hàng, bạn có thể thấy trung bình doanh thu tăng vào cuối tuần, hoặc khách hàng từ thành phố lớn chi tiêu nhiều hơn 20% so với vùng nông thôn.
## 3.Trực quan hóa dữ liệu (Data Visualization)
- là cách biến dữ liệu thành hình ảnh để dễ hiểu và truyền đạt thông tin hiệu quả hơn. Đây là bước kết nối giữa phân tích và giao tiếp.
    - Biểu đồ cơ bản:
        - Histogram: Hiển thị phân bố của một biến, như phân bố tuổi của khách hàng.
        - Boxplot: Phát hiện giá trị bất thường và so sánh phân bố giữa các nhóm (ví dụ: chi tiêu của nam vs nữ).
        - Scatter plot: Xem mối quan hệ giữa hai biến, như "tuổi" và "số tiền chi tiêu".
    - Biểu đồ nâng cao:
        -Heatmap: Hiển thị ma trận tương quan giữa các biến, giúp phát hiện mối liên hệ mạnh/yếu.
        -Line chart: Theo dõi xu hướng theo thời gian, như doanh thu hàng tháng.
        - Bar chart: So sánh giá trị giữa các nhóm, như doanh thu theo khu vực.
    - Nguyên tắc thiết kế: Đảm bảo biểu đồ rõ ràng, không quá tải thông tin, sử dụng màu sắc hợp lý để làm nổi bật điểm chính.
    - Công cụ phổ biến: Python (Matplotlib, Seaborn), R (ggplot2), hoặc các phần mềm như Tableau, Power BI
- Ví dụ thực tế: Bạn vẽ một histogram và thấy 80% khách hàng dưới 35 tuổi, hoặc dùng heatmap để phát hiện "số lần mua hàng" có tương quan mạnh với "số tiền chi tiêu".
## Tổng kết
- EDA là quá trình khám phá dữ liệu một cách có hệ thống để hiểu rõ bản chất của nó, phát hiện các mẫu hình, mối quan hệ và bất thường, đồng thời trình bày kết quả một cách trực quan.
    - Hiểu dữ liệu giúp bạn làm quen với "nguyên liệu thô".
    - Phân tích dữ liệu đào sâu để tìm insight.
    - Trực quan hóa dữ liệu biến insight thành câu chuyện dễ hiểu.