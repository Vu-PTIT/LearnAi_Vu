# I. Tổng quan về các thuật toán tổ hợp trong học máy

## 1. Bagging và Random Forest

### a.Bagging (Bootstrap Aggregating)

**Nguyên lý hoạt động:**

- Bagging là kỹ thuật tổ hợp nhằm giảm phương sai (variance) và ngăn ngừa overfitting bằng cách huấn luyện nhiều mô hình con trên các tập dữ liệu con được lấy mẫu ngẫu nhiên từ tập dữ liệu gốc (có hoàn lại). Dự đoán cuối cùng được đưa ra bằng cách trung bình (đối với hồi quy) hoặc bỏ phiếu đa số (đối với phân loại) từ các mô hình con.

**Quy trình huấn luyện:**

- **Lấy mẫu dữ liệu:** Tạo nhiều tập dữ liệu con bằng cách lấy mẫu ngẫu nhiên có hoàn lại từ tập dữ liệu gốc.
- **Huấn luyện mô hình con:** Huấn luyện một mô hình con (thường là cây quyết định) trên mỗi tập dữ liệu con.
- **Tổng hợp dự đoán:** Kết hợp dự đoán từ các mô hình con bằng cách trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại).

**Đặc điểm nổi bật:**

- Giảm phương sai và overfitting.
- Hiệu quả với dữ liệu có nhiễu.
- Dễ dàng triển khai và song song hóa.

###  b.Random Forest(Rừng ngẫu nhiên)

**Nguyên lý hoạt động:**

- Random Forest là một biến thể của Bagging, sử dụng nhiều cây quyết định (decision trees) làm mô hình con. Mỗi cây được huấn luyện trên một tập dữ liệu con và tại mỗi nút phân chia, chỉ một tập con ngẫu nhiên của các đặc trưng được xem xét để tạo ra sự đa dạng giữa các cây, giúp giảm thiểu overfitting và cải thiện độ chính xác của mô hình.

**Quy trình huấn luyện:**

- **Tạo tập dữ liệu con:** Sử dụng kỹ thuật bagging để tạo nhiều tập dữ liệu con.
- **Huấn luyện cây quyết định:** Huấn luyện một cây quyết định trên mỗi tập dữ liệu con, với việc chọn ngẫu nhiên một tập con của các đặc trưng tại mỗi nút phân chia.
- **Tổng hợp dự đoán:** Kết hợp dự đoán từ các cây bằng cách trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại).

**Đặc điểm nổi bật:**

- Giảm phương sai và overfitting.
- Hiệu quả với dữ liệu có nhiễu.
- Dễ dàng triển khai và song song hóa.

## 2. Boosting: AdaBoost và XGBoost

###  a.AdaBoost (Adaptive Boosting)

**Nguyên lý hoạt động:**

AdaBoost là thuật toán Boosting đầu tiên, hoạt động bằng cách huấn luyện các mô hình con tuần tự, trong đó mỗi mô hình mới cố gắng sửa lỗi của mô hình trước đó. Các mô hình con thường là các "người học yếu" (weak learners), như cây quyết định nông. AdaBoost điều chỉnh trọng số của các mẫu dữ liệu: tăng trọng số cho các mẫu bị phân loại sai để mô hình tiếp theo tập trung vào chúng.

**Quy trình huấn luyện:**

- **Khởi tạo trọng số:** Gán trọng số bằng nhau cho tất cả các mẫu dữ liệu.
- **Huấn luyện mô hình con:** Huấn luyện một mô hình con trên tập dữ liệu với trọng số hiện tại.
- **Tính toán lỗi:** Tính toán lỗi của mô hình con trên tập dữ liệu.
- **Tính toán trọng số mô hình:** Tính toán trọng số của mô hình con dựa trên lỗi.
- **Cập nhật trọng số mẫu:** Tăng trọng số cho các mẫu bị phân loại sai và giảm trọng số cho các mẫu được phân loại đúng.
- **Lặp lại:** Lặp lại các bước trên cho đến khi đạt được số lượng mô hình con mong muốn hoặc lỗi giảm xuống dưới ngưỡng cho phép.

**Đặc điểm nổi bật:**

- Giảm độ chệch và cải thiện độ chính xác.
- Hiệu quả với dữ liệu không cân bằng.
- Dễ bị overfitting nếu không điều chỉnh đúng.

### b.XGBoost (Extreme Gradient Boosting)

**Nguyên lý hoạt động:**

XGBoost là một phiên bản nâng cao của Boosting, sử dụng kỹ thuật tối ưu hóa gradient và regularization để cải thiện hiệu suất và ngăn ngừa overfitting. XGBoost nổi tiếng với hiệu suất cao và khả năng xử lý dữ liệu lớn.

**Quy trình huấn luyện:**

- **Khởi tạo mô hình:** Bắt đầu với một mô hình đơn giản.
- **Tính toán gradient:** Tính toán gradient của hàm mất mát đối với dự đoán hiện tại.
- **Huấn luyện mô hình con:** Huấn luyện một mô hình con để dự đoán gradient.
- **Cập nhật mô hình:** Cập nhật mô hình bằng cách cộng mô hình con mới với một hệ số học.
- **Lặp lại:** Lặp lại các bước trên cho đến khi đạt được số lượng mô hình con mong muốn hoặc lỗi giảm xuống dưới ngưỡng cho phép.

**Đặc điểm nổi bật:**

- Giảm độ chệch và cải thiện độ chính xác.
- Hiệu quả với dữ liệu không cân bằng.
- Có khả năng xử lý dữ liệu lớn và hỗ trợ regularization.

## 3. Stacking (Stacked Generalization)

**Nguyên lý hoạt động:**

Stacking là kỹ thuật tổ hợp kết hợp nhiều mô hình con (có thể khác nhau về loại) bằng cách sử dụng một mô hình meta (meta-model) để học cách kết hợp dự đoán từ các mô hình con.

**Quy trình huấn luyện:**

- **Tầng cơ sở (base level):** Huấn luyện các mô hình con trên tập dữ liệu gốc.
- **Tạo tập dữ liệu meta:** Sử dụng dự đoán của các mô hình con trên tập validation để tạo tập dữ liệu meta.
- **Tầng meta (meta level):** Huấn luyện mô hình meta trên tập dữ liệu meta.
- **Dự đoán:** Để dự đoán cho dữ liệu mới, sử dụng các mô hình con để tạo đặc trưng và đưa vào mô hình meta để dự đoán cuối cùng.

**Đặc điểm nổi bật:**

- Khai thác sức mạnh của nhiều mô hình khác nhau.
- Thường cho hiệu suất cao hơn so với các mô hình đơn lẻ.
- Phức tạp trong việc triển khai và điều chỉnh.

## 4.  Voting Classifier

**Nguyên lý hoạt động:**

Voting Classifier là kỹ thuật tổ hợp kết hợp dự đoán từ nhiều mô hình con bằng cách bỏ phiếu. Có hai loại chính:

- **Hard Voting:** Dự đoán cuối cùng là lớp được nhiều mô hình con dự đoán nhất.
- **Soft Voting:** Dự đoán cuối cùng dựa trên trung bình xác suất dự đoán từ các mô hình con.

**Quy trình huấn luyện:**

- **Huấn luyện mô hình con:** Huấn luyện các mô hình con trên tập dữ liệu gốc.
- **Dự đoán:** Dự đoán cho dữ liệu mới bằng cách kết hợp kết quả từ các mô hình con bằng hard voting hoặc soft voting.

**Đặc điểm nổi bật:**

- Dễ triển khai và hiểu.
- Cải thiện độ chính xác bằng cách kết hợp nhiều mô hình.
- Hiệu suất phụ thuộc vào chất lượng và sự đa dạng của các mô hình con.

# II. Semi-Supervised Learning (Học bán giám sát)

## 1. Khái niệm

**Semi-Supervised Learning (SSL)** là một phương pháp học máy kết hợp giữa **học có giám sát** (supervised learning) và **học không giám sát** (unsupervised learning).
Trong SSL, mô hình được huấn luyện trên một tập dữ liệu bao gồm cả dữ liệu có nhãn và dữ liệu không có nhãn.
Mục tiêu là tận dụng thông tin từ dữ liệu không nhãn để cải thiện hiệu suất của mô hình, đặc biệt khi dữ liệu có nhãn khan hiếm hoặc khó thu thập.

## 2. Ứng dụng khi dữ liệu nhãn hạn chế

SSL đặc biệt hữu ích trong các tình huống sau:

- **Phân loại hình ảnh y tế**: Gán nhãn cho hình ảnh y tế đòi hỏi chuyên môn cao và tốn kém thời gian. SSL giúp tận dụng dữ liệu không nhãn để cải thiện mô hình chẩn đoán.

- **Xử lý ngôn ngữ tự nhiên**: Trong các bài toán như phân loại văn bản hoặc phân tích cảm xúc, việc gán nhãn dữ liệu có thể tốn kém. SSL cho phép sử dụng dữ liệu văn bản không nhãn để nâng cao hiệu suất mô hình.

- **Phát hiện gian lận**: Trong lĩnh vực tài chính, dữ liệu về các giao dịch gian lận thường rất ít. SSL giúp mô hình học từ dữ liệu không nhãn để phát hiện các mẫu gian lận mới.

## 3. Các phương pháp cơ bản

### a. Pseudo-Labeling

Phương pháp này gán nhãn giả cho dữ liệu không nhãn dựa trên dự đoán của mô hình hiện tại.
Sau đó, dữ liệu có nhãn giả được sử dụng cùng với dữ liệu có nhãn thật để huấn luyện mô hình.
Quá trình này có thể được lặp lại nhiều lần để cải thiện độ chính xác của mô hình.

### b. Self-Training

Self-Training là một quy trình lặp đi lặp lại, trong đó:

1. Huấn luyện mô hình trên dữ liệu có nhãn.
2. Sử dụng mô hình để dự đoán nhãn cho dữ liệu không nhãn.
3. Chọn các mẫu có độ tin cậy cao và thêm chúng vào tập dữ liệu có nhãn.
4. Lặp lại quá trình với tập dữ liệu mở rộng.

### c. Co-Training

Co-Training sử dụng hai mô hình học khác nhau, mỗi mô hình học trên một tập đặc trưng khác nhau của dữ liệu.
Mỗi mô hình gán nhãn cho dữ liệu không nhãn và chia sẻ các nhãn này với mô hình kia để cải thiện hiệu suất tổng thể.

### d. Graph-Based Label Propagation

Phương pháp này xây dựng một đồ thị trong đó các nút đại diện cho các mẫu dữ liệu và các cạnh thể hiện mối quan hệ giữa các mẫu.
Nhãn từ các mẫu có nhãn được lan truyền qua đồ thị để gán nhãn cho các mẫu không nhãn dựa trên sự tương đồng.

## 4. Ưu điểm và thách thức

### Ưu điểm

- Giảm chi phí và thời gian gán nhãn dữ liệu.
- Tận dụng được lượng lớn dữ liệu không nhãn có sẵn.
- Cải thiện hiệu suất mô hình khi dữ liệu có nhãn hạn chế.

### Thách thức

- Chất lượng dữ liệu không nhãn có thể ảnh hưởng đến hiệu suất mô hình.
- Khó khăn trong việc xác định độ tin cậy của nhãn giả.
- Cần thiết kế cẩn thận để tránh việc mô hình học sai lệch từ dữ liệu không nhãn.

# III. Probabilistic Graphical Models (PGM)

## 1. Khái niệm cơ bản

Probabilistic Graphical Models (PGM) là mô hình kết hợp giữa lý thuyết đồ thị và xác suất thống kê để biểu diễn mối quan hệ ngẫu nhiên giữa các biến trong hệ thống phức tạp.

Có hai dạng chính:

### a. Bayesian Networks (Mạng Bayes)

- Mô hình đồ thị có hướng (Directed Acyclic Graph - DAG).
- Các nút biểu diễn các biến ngẫu nhiên.
- Các cung thể hiện mối quan hệ nhân quả có điều kiện.
- Mỗi nút gắn với một bảng phân phối xác suất có điều kiện (CPT - Conditional Probability Table).

**Ví dụ:** Mô hình chẩn đoán bệnh gồm các nút như triệu chứng, bệnh, xét nghiệm và mối quan hệ giữa chúng cho phép suy luận nguyên nhân từ các quan sát.

### b. Markov Networks (Markov Random Fields)

- Mô hình đồ thị vô hướng (Undirected Graph).
- Biểu diễn sự phụ thuộc giữa các biến qua các "clique" (nhóm biến liên thông).
- Mỗi clique gắn với một hàm thế (potential function) dùng để tính xác suất toàn cục.

#### So sánh nhanh

| Đặc điểm       | Bayesian Network | Markov Network |
|----------------|------------------|----------------|
| Loại đồ thị     | Có hướng          | Vô hướng       |
| Biểu diễn       | Nhân quả          | Quan hệ phụ thuộc |
| Đặc trưng       | CPT               | Potential Function |


## 2. Ứng dụng của PGM trong Machine Learning

### a. Suy luận (Inference)

- Dự đoán giá trị chưa biết từ giá trị quan sát.
- Suy diễn tiến (forward) và ngược (backward).

### b. Học cấu trúc và tham số

- Học cấu trúc đồ thị từ dữ liệu.
- Ước lượng các bảng CPT hoặc hàm thế.

### c. Xử lý ngôn ngữ tự nhiên (NLP)

- Gán nhãn từ loại, phân tích cú pháp, nhận diện thực thể.

### d. Computer Vision

- Phân đoạn ảnh, phát hiện đối tượng, phục hồi ảnh.

### e. Robotics và điều khiển

- Lập kế hoạch trong môi trường không chắc chắn.
- Lọc và theo dõi trạng thái như Bayes Filter.

PGM là một công cụ mạnh trong Machine Learning khi cần biểu diễn rõ ràng và hiệu quả các mối quan hệ xác suất giữa các biến trong hệ thống phức tạp.
# IV. Recommendation Systems

## 1. Các phương pháp cơ bản

Recommendation Systems (hệ thống gợi ý) là công nghệ giúp cá nhân hóa trải nghiệm người dùng bằng cách đề xuất các mặt hàng, nội dung, hoặc dịch vụ phù hợp với sở thích hoặc hành vi của người dùng.

### a. Content-Based Filtering

- **Nguyên lý:** Gợi ý các mục (items) tương tự với những gì người dùng đã thích trong quá khứ, dựa trên đặc trưng nội dung của mục đó.
- **Dữ liệu sử dụng:** Đặc điểm của sản phẩm (mô tả văn bản, thể loại, tác giả, diễn viên,...).
- **Mô hình hóa:** Tính toán mức độ tương đồng giữa các mục dựa trên đặc trưng, ví dụ dùng cosine similarity giữa vectors.

**Ưu điểm:**
- Cá nhân hóa tốt.
- Không cần dữ liệu từ người dùng khác.

**Hạn chế:**
- Không khám phá được sở thích mới (đề xuất thiên về nội dung quen thuộc).
- Phụ thuộc vào chất lượng mô tả đặc trưng nội dung.

### b. Collaborative Filtering

- **Nguyên lý:** Dựa trên hành vi tương tự của người dùng hoặc sự đánh giá tương tự của các mục.
- **Hai loại chính:**
  - **User-based:** Gợi ý mục mà người dùng tương tự đã thích.
  - **Item-based:** Gợi ý mục tương tự với những gì người dùng hiện tại đã thích.

- **Mô hình hóa:** Dựa vào ma trận người dùng – mục (user-item matrix), có thể dùng các kỹ thuật như:
  - Matrix Factorization (SVD, ALS)
  - k-NN (k-nearest neighbors)

**Ưu điểm:**
- Không cần đặc trưng nội dung.
- Tự động khám phá mối quan hệ ẩn giữa người dùng và mục.

**Hạn chế:**
- Cold start problem (người dùng mới, mục mới).
- Ma trận thưa (sparse matrix) làm giảm độ chính xác.

## 2. Ứng dụng thực tế

### Netflix

- Kết hợp Collaborative Filtering và Content-Based để đề xuất phim dựa trên lịch sử xem và đánh giá của người dùng.
- Sử dụng Deep Learning để phân tích metadata (tên phim, thể loại, diễn viên) và hành vi người dùng.

### Amazon

- Gợi ý sản phẩm dựa trên lịch sử mua sắm, sản phẩm đã xem, giỏ hàng, và sản phẩm tương tự đã mua bởi người dùng khác.
- Dùng cả Collaborative Filtering (ví dụ: "người dùng mua A cũng mua B") và Content-Based Filtering (ví dụ: sản phẩm tương tự).

### Spotify

- Sử dụng mô hình học sâu để gợi ý bài hát dựa trên sở thích nghe nhạc, hành vi tương tác, cũng như đặc điểm âm thanh (audio features).
- Hệ thống gợi ý "Discover Weekly" dựa vào collaborative filtering và phân tích nội dung bài hát.

# V. Giới thiệu MLOps

## 1. Khái niệm MLOps và lý do cần MLOps

**MLOps (Machine Learning Operations)** là tập hợp các thực hành kết hợp giữa **Machine Learning** và **DevOps** nhằm mục tiêu tự động hóa và quản lý toàn bộ vòng đời của mô hình học máy.

### Tại sao cần MLOps?

- Đảm bảo triển khai mô hình ML vào môi trường sản xuất một cách ổn định và lặp lại được.
- Hạn chế rủi ro khi mô hình hoạt động không hiệu quả do thay đổi dữ liệu (data drift, concept drift).
- Tăng tốc độ phát triển và cập nhật mô hình.
- Hỗ trợ theo dõi, kiểm thử, và tái sử dụng mô hình.

## 2. ML Pipeline là gì?

**ML Pipeline** là chuỗi các bước cần thiết để xây dựng, triển khai và duy trì mô hình học máy.

### Thành phần cơ bản:
- Thu thập và xử lý dữ liệu.
- Trích xuất đặc trưng (Feature Engineering).
- Huấn luyện và đánh giá mô hình.
- Triển khai mô hình.
- Giám sát và tái huấn luyện mô hình.

## 3. Các công cụ phổ biến trong MLOps

| Công cụ | Mục đích chính |
|--------|----------------|
| **MLflow** | Theo dõi thí nghiệm, quản lý model và triển khai |
| **Kubeflow** | Xây dựng ML pipeline trên nền Kubernetes |
| **DVC** | Quản lý phiên bản dữ liệu và pipeline |
| **TFX (TensorFlow Extended)** | Triển khai pipeline TensorFlow quy mô lớn |
| **Seldon / BentoML** | Triển khai mô hình ML như dịch vụ |
| **Evidently / Prometheus / Grafana** | Giám sát hiệu suất và cảnh báo |

## 4. Các bước chính trong pipeline MLOps

### 1. Model Training
- Thu thập và xử lý dữ liệu.
- Huấn luyện mô hình trên dữ liệu sạch và đầy đủ.
- Theo dõi hiệu suất và lưu version mô hình.

### 2. Deployment
- Đóng gói mô hình dưới dạng API hoặc container.
- Triển khai lên hệ thống thật (server, cloud...).

### 3. Monitoring
- Theo dõi hiệu suất và chất lượng mô hình trong sản xuất:
  - Data drift
  - Accuracy degradation
  - Latency, throughput...

### 4. Retraining
- Khi phát hiện hiệu suất mô hình suy giảm.
- Tự động thu thập dữ liệu mới, huấn luyện lại và triển khai phiên bản mới.

MLOps giúp các tổ chức triển khai ML một cách tin cậy, có thể mở rộng và duy trì mô hình dễ dàng trong suốt vòng đời của nó.