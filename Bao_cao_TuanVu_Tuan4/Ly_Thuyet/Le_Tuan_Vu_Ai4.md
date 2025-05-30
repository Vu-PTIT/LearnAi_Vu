# I. Supervised và Unsupervised learning
## A.Supervised Learning
### 1. Định nghĩa
  - Học có giám sát là một loại học máy trong đó một mô hình được đào tạo trên dữ liệu được dán nhãn, có nghĩa là mỗi đầu vào được ghép nối với đầu ra chính xác.
### 2. Mục tiêu
  - Mô hình học bằng cách so sánh dự đoán của nó với các câu trả lời thực tế được cung cấp trong dữ liệu đào tạo. 
  - Theo thời gian, nó tự điều chỉnh để giảm thiểu lỗi và cải thiện độ chính xác. Mục tiêu của việc học có giám sát là đưa ra dự đoán chính xác khi được cung cấp dữ liệu mới, không nhìn thấy. 
  - Ví dụ: nếu một mô hình được đào tạo để nhận ra các chữ số viết tay, nó sẽ sử dụng những gì nó đã học để xác định chính xác các số mới mà nó đã thấy trước đó.
### 3. Hoạt động
   - Quá trình này hoạt động thông qua:
       - Training Data: Mô hình được cung cấp với bộ dữ liệu đào tạo bao gồm dữ liệu đầu vào (features) và dữ liệu đầu ra tương ứng (labels).
       - Learning Process: Thuật toán xử lý dữ liệu đào tạo, học các mối quan hệ giữa các features đầu vào và labels đầu ra. Điều này đạt được bằng cách điều chỉnh các tham số mô hình để giảm thiểu sự khác biệt giữa các dự đoán của nó và các nhãn thực tế.
   - Sau khi đào tạo, mô hình được đánh giá bằng cách sử dụng test dataset để đo độ chính xác và hiệu suất của nó. Sau đó, hiệu suất của mô hình được tối ưu hóa bằng cách điều chỉnh các tham số và sử dụng các kỹ thuật để cân bằng bias và variance. Điều này đảm bảo mô hình khái quát hóa tốt cho dữ liệu mới.  
   - Quy trình :
     - Data Collection and Preprocessing: Thu thập một bộ dữ liệu được dán nhãn bao gồm các tính năng đầu vào và nhãn đầu ra đích. Làm sạch dữ liệu, xử lý các giá trị bị thiếu và các tính năng quy mô khi cần thiết để đảm bảo chất lượng cao cho các thuật toán .
     - Splitting the Data: chia dữ liệu thành training set (80%) and the test set (20%).
     - Choosing the Model: Chọn thuật toán thích hợp dựa trên loại vấn đề. 
     - Training the Model: Cung cấp dữ liệu đầu vào mô hình và nhãn đầu ra, cho phép nó tìm hiểu các mẫu bằng cách điều chỉnh các tham số bên trong.
     - Evaluating the Model: Kiểm tra mô hình được đào tạo trên bộ thử nghiệm chưa từng thấy và đánh giá hiệu suất của nó bằng các số liệu khác nhau.
     - Hyperparameter Tuning: Điều chỉnh cài đặt kiểm soát quá trình đào tạo (ví dụ: tốc độ học tập) bằng các kỹ thuật như grid và cross-validation
     - Final Model Selection and Testing: Huấn luyện lại trên bộ dữ liệu hoàn chỉnh bằng cách sử dụng các siêu dự án tốt nhất để kiểm tra hiệu suất của nó trên bộ thử nghiệm để đảm bảo sự sẵn sàng để triển khai.
     - Model Deployment: Triển khai mô hình được xác thực để đưa ra dự đoán về dữ liệu mới, không nhìn thấy.
### 4. Phân biệt
   - Supervised learning có thể được áp dụng cho hai loại vấn đề chính:
     - Classification(Phân loại): các biến output là các biến phân loại(ví dụ:  email spam hay không, yes hay no)
     - Regression(Hồi quy): các biến outout là các biến liên tục(ví dụ: dự đoán giá nhà, giá cổ phiếu).  

| Tiêu chí            | Classification                                         | Regression                                                     |
| ------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| Output              | Các giá trị phân loại                                  | Các giá trị liên tục                                           |
| Mục đích            | Gán dữ liệu vào các lớp cụ thể                         | Dự đoán giá trị số lượng dựa trên dữ liệu đầu vào              |
| Thuật toán phổ biến | Logistic Regression, Decision Tree, SVM, Random Forest | Random Forest	Linear Regression, Decision Tree Regression, SVR |
| Đánh giá mô hình    | Accuracy, Precision, Recall, F1-score, ROC-AUC         | RMSE, MAE, R² Score                                            |

## B.Unspuervised Learning
### 1. Định nghĩa
  - Học không giám sát là một loại học máy trong đó một mô hình được đào tạo trên liên quan đến dữ liệu không nhãn
### 2. Mục tiêu
  - Các thuật toán  được giao nhiệm vụ tìm kiếm các mẫu và mối quan hệ trong dữ liệu mà không có bất kỳ kiến ​​thức trước nào về ý nghĩa của dữ liệu. 
  - Các thuật toán  tìm thấy các mẫu và dữ liệu ẩn mà không có bất kỳ sự can thiệp nào của con người, tức là, chúng ta không cung cấp đầu ra cho mô hình của chúng ta. Mô hình đào tạo chỉ có các giá trị tham số đầu vào và tự mình phát hiện ra các nhóm hoặc mẫu
### 3. Hoạt động
  - Mô hình học bằng cách phân tích dữ liệu không nhãn để xác định các mẫu và mối quan hệ. Dữ liệu không được dán nhãn với bất kỳ danh mục hoặc kết quả được xác định trước, vì vậy thuật toán phải tự mình tìm các mẫu và mối quan hệ này. 
  - Đây có thể là một nhiệm vụ đầy thách thức, nhưng nó cũng có thể rất bổ ích, vì nó có thể tiết lộ những hiểu biết sâu sắc về dữ liệu sẽ không rõ ràng từ một bộ dữ liệu được dán nhãn.
  - Đầu vào cho unsupervised learning models như sau: 
    - Unstructured data: có thể chứa dữ liệu noisy (meaningless), các giá trị bị thiếu hoặc dữ liệu không xác định
    - Unlabeled data: Dữ liệu chỉ chứa giá trị cho các tham số đầu vào, không có giá trị được nhắm mục tiêu (output). 
### 4. Phấn biệt
- Có 3 loại thuật toán chính được sử dụng :
  - Clustering(Phân cụm)
  - Association Rule Learning(ARL)
  - Dimensionality Reduction(Giảm chiều)

| Tiêu chí    | Association Rule Learning (ARL)                                                       | Clustering (Phân cụm)                                                              | Dimensionality Reduction (Giảm chiều)                                                                 |
| ----------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Mục tiêu    | Tìm kiếm các mối quan hệ hoặc luật kết hợp giữa các biến trong dữ liệu.               | Nhóm các điểm dữ liệu tương tự nhau vào cùng một cụm.                              | Giảm số lượng đặc trưng trong dữ liệu, giữ lại thông tin quan trọng nhất.                             |
| Đầu ra      | Các luật kết hợp dạng "Nếu X thì Y" với các chỉ số như Support, Confidence, Lift.     | Nhãn cụm cho mỗi điểm dữ liệu (ví dụ: cụm 1, cụm 2, ...).                          | Dữ liệu mới với số chiều thấp hơn, mỗi điểm dữ liệu được biểu diễn trong không gian đặc trưng mới.    |
| Ứng dụng    | Phân tích giỏ hàng, phân tích hành vi người dùng, phát hiện gian lận, y học.          | Phân đoạn thị trường, phân nhóm khách hàng, phát hiện cộng đồng trong mạng xã hội. | Trực quan hóa dữ liệu cao chiều, giảm nhiễu, tăng hiệu suất mô hình học máy.                          |
| Thuật toán  | Apriori, Eclat, FP-Growth.                                                            | K-Means, DBSCAN, Hierarchical Clustering.                                          | PCA (Principal Component Analysis), t-SNE (t-Distributed Stochastic Neighbor Embedding), Autoencoder. |
| Mối quan hệ | Tập trung vào việc tìm kiếm mối quan hệ giữa các biến mà không cần phân nhóm dữ liệu. | Tập trung vào việc phân nhóm dữ liệu mà không tìm kiếm mối quan hệ giữa các biến.  | Tập trung vào việc giảm số chiều của dữ liệu để đơn giản hóa mô hình hoặc trực quan hóa dữ liệu.      |

# II. So sánh Supervised Learning và Unsupervised Learning

| Tiếu chí           | Supervised Learning                                                                                     | Unsupervised Learning                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Dữ liệu            | Có nhãn: dữ liệu đầu vào kèm đầu ra (nhãn)​                                                             | Không nhãn: chỉ có dữ liệu đầu vào, mô hình tự khám phá mẫu/cấu trúc ẩn​                                   |
| Mục tiêu           | Dự đoán kết quả cụ thể (phân loại/hồi quy) dựa trên mối quan hệ input–output đã học​                    | Khám phá các mẫu hoặc cấu trúc tiềm ẩn trong dữ liệu đầu vào chưa gán nhãn                                 |
| Ứng dụng           | dụng	Ví dụ: phân loại thư rác, nhận diện hình ảnh, dự báo giá (cổ phiếu, thời tiết…)​                   | Ví dụ: phân nhóm khách hàng, hệ thống gợi ý (recommendation), phát hiện bất thường, phân tích dữ liệu lớn​ |
| Độ phức tạp        | Huấn luyện có hướng dẫn (có nhãn) nên mô hình hội tụ nhanh hơn​; đòi hỏi nhiều nhãn (tốn công gán nhãn) | Huấn luyện tự động trên dữ liệu lớn (không nhãn) nên tốn kém tài nguyên tính toán                          |
| Đánh giá           | Accuracy, Precision, Recall, F1-Score, MSE/MAE                                                          | , Davies–Bouldin, Calinski–Harabasz…​                                                                      |
| Phụ thuộc vào nhãn | Rất cao: cần dữ liệu được gán nhãn để huấn luyện​                                                       | Thấp/Không: không yêu cầu nhãn, hoạt động trực tiếp trên dữ liệu thô​                                      |

# III. Các thuật toán
### 1. Supervised Learning  
- [Linear Regression (Hồi quy tuyến tính)](https://en.wikipedia.org/wiki/Linear_regression)
  - Mục tiêu: Xây dựng mô hình tuyến tính để dự đoán giá trị liên tục của biến phụ thuộc (VD: giá nhà, doanh thu) dựa trên một hoặc nhiều biến độc lập. Cụ thể, hồi quy tuyến tính ước lượng mối quan hệ tuyến tính giữa biến đáp ứng (phụ thuộc) và các biến giải thích (độc lập)​
  - Nguyên lý hoạt động: Mô hình giả sử hàm trung bình có điều kiện của biến đáp ứng là một hàm tuyến tính của biến giải thích. Các hệ số của mô hình được ước lượng (thường bằng phương pháp bình phương tối thiểu – OLS) sao cho tổng bình phương sai số giữa giá trị dự đoán và giá trị thực nhỏ nhất
  - Ứng dụng: Hồi quy tuyến tính được dùng rộng rãi trong dự đoán và phân tích xu hướng. Ví dụ: dự báo giá bất động sản, dự báo doanh thu, phân tích xu hướng thời gian (trend line), mô hình tài chính như CAPM (beta của cổ phiếu), kinh tế học (dự báo GDP, tiêu dùng)
- [Decision Tree (Cây quyết định)](https://en.wikipedia.org/wiki/Decision_tree_learning)
  - Mục tiêu: Học cây quyết định để phân loại (classification) hoặc hồi quy (regression). Mỗi nút trong cây tương ứng với một điều kiện trên một đặc trưng, tách dữ liệu thành các nhóm con sao cho ở các lá cây thuần nhất về giá trị mục tiêu. Mục tiêu là xây dựng mô hình dự đoán biến mục tiêu từ các đặc trưng đầu vào​
  - Nguyên lý hoạt động: Xây dựng cây quyết định theo hướng phân tách đệ quy (“top-down”). Từ tập dữ liệu gốc (gốc cây), thuật toán chọn một đặc trưng và ngưỡng tách sao cho độ đồng nhất (độ “sạch” về nhãn lớp hoặc giá trị) của các nhóm con được cải thiện nhất. Quá trình tiếp tục trên từng nhóm con cho đến khi đạt điều kiện dừng (ví dụ không thể tách thêm hoặc nhóm con đã đồng nhất). Kết quả là một cây nhị phân (hoặc đa nhánh) mà các nút lá dự đoán nhãn hoặc giá trị mục tiêu
  - Ứng dụng: Cây quyết định được dùng nhiều trong phân tích dữ liệu, đặc biệt khi cần giải thích kết quả. Ví dụ: phân loại khách hàng theo rủi ro tín dụng, dự đoán tỷ lệ giữ chân khách hàng, chẩn đoán y tế, ước tính giá nhà (quyết định phân nhánh theo đặc điểm nhà), và còn dùng làm thành phần cơ bản trong các mô hình phức tạp (ví dụ rừng ngẫu nhiên - Random Forest).
- [Logistic Regression (Hồi quy logistic)](https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/)
  - Mục tiêu: Phân loại nhị phân (binary classification) – dự đoán xác suất một biến mục tiêu thuộc lớp 0 hoặc 1 (ví dụ: mắc bệnh hay không, email spam hay không). Logistic Regression mở rộng được cho phân loại đa lớp (multinomial) khi cần​
  - Nguyên lý hoạt động: Mô hình học một kết hợp tuyến tính từ biến đầu vào để ước lượng log-odds (log(p/(1-p))) của biến nhị phân, sau đó đưa qua hàm sigmoid (hàm logistic) để ra xác suất. Hay nói cách khác, đầu ra của hàm tuyến tính được biến đổi bởi hàm sigmoid để đảm bảo giá trị dự đoán nằm trong [0,1]​
  - Ứng dụng: Logistic Regression rất phổ biến trong bài toán phân loại nhị phân. Ví dụ: chẩn đoán y khoa (bệnh tật hay không), lọc thư rác, phân loại tín dụng (phê duyệt hay từ chối), dự đoán khả năng nhấp chuột vào quảng cáo (yes/no), các bài toán phân loại nhãn cơ bản trong marketing và tài chính
### 2. Unsupervised Learning  
- [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
  - Mục tiêu: Phân cụm dữ liệu thành k nhóm (clusters) sao cho các điểm trong cùng cụm gần nhau nhất về khoảng cách (thông thường là Euclid), và xa các cụm khác nhất. Mỗi nhóm được đặc trưng bởi một tâm cụm (centroid)​
  - Nguyên lý hoạt động: Thuật toán Lloyd (naïve k-means) khởi tạo k tâm cụm ngẫu nhiên, sau đó lặp lại hai bước: (1) Gán mỗi điểm dữ liệu cho cụm có tâm gần nó nhất; (2) Cập nhật tâm cụm mới bằng cách tính trung bình tọa độ các điểm được gán. Lặp lại đến khi không còn thay đổi (các cụm hội tụ) hoặc đạt tới số vòng lặp tối đa. Mục tiêu tối ưu là giảm tổng phương sai nội cụm (WCSS).
  - Ứng dụng: K-Means được dùng phổ biến trong phân tích thị trường (phân nhóm khách hàng), xử lý ảnh (nén ảnh, phân vùng ảnh), hệ gợi ý (lấy nhóm người tương tự), giảm dữ liệu và tiền xử lý trước khi áp dụng các thuật toán khác. Nó cũng là nền tảng cho các phương pháp gom cụm khác và trong nghiên cứu học máy như biểu diễn tập tính năng.  
- [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](https://en.wikipedia.org/wiki/DBSCAN)
  - Mục tiêu: Phân cụm dựa trên mật độ: nhóm các điểm có mật độ cao liên tiếp thành cụm và đánh dấu các điểm thưa như nhiễu (noise). Không cần biết trước số cụm; tự động xác định vùng có mật độ dày và khu vực nhiễu.
  - Nguyên lý hoạt động: Thuật toán định nghĩa hai tham số chính: epsilon (bán kính lân cận) và MinPts (số điểm tối thiểu trong bán kính). Điểm được coi là core point nếu trong bán kính epsilon có ≥ MinPts điểm khác. Các core point kết nối với nhau tạo thành cụm; các điểm chỉ kề cận (border points) được gán vào cụm lân cận; điểm không thuộc core hay border sẽ bị đánh dấu là noise. Phân cụm kết quả dựa trên mật độ các điểm xung quanh.
  - Ứng dụng: DBSCAN thường dùng trong phát hiện bản mẫu mật độ, như phát hiện điểm bất thường (outliers), phân cụm hình ảnh, GIS (gần nhau về địa lý), xử lý dữ liệu địa không gian, và bất kỳ bài toán gom cụm nào mà cấu trúc cụm có hình dáng và mật độ tự nhiên.  
- [PCA (Principal Component Analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis)
  - Mục tiêu: Giảm chiều dữ liệu bằng cách tìm các hướng chính (principal components) – những tổ hợp tuyến tính của đặc trưng gốc sao cho giữ lại tối đa phương sai của dữ liệu. Mục đích là giảm số biến mà vẫn bảo toàn hầu hết thông tin (phương sai) của dữ liệu gốc
  - Nguyên lý hoạt động: PCA thực hiện phép biến đổi tuyến tính để chuyển dữ liệu sang hệ tọa độ mới (các thành phần chính). Thành phần chính thứ nhất (PC1) là trục (hướng) trong không gian đặc trưng mà phương sai của dữ liệu chiếu lên lớn nhất. Thành phần chính thứ hai (PC2) là hướng trực giao với PC1 có phương sai lớn nhất còn lại, và tiếp tục như vậy. Toàn bộ quá trình tương đương với tính eigenvector của ma trận hiệp phương sai dữ liệu (hoặc thực hiện phân tích SVD). Các thành phần chính này là tập hợp hệ quy chiếu mới trong đó các chiều của dữ liệu không tương quan tuyến tính với nhau.
  - Ứng dụng: PCA được dùng trong mọi lĩnh vực cần giảm chiều hoặc trực quan hóa dữ liệu: phân tích dữ liệu thăm dò, hình ảnh y sinh (di truyền), khí tượng, tài chính (phân tích rủi ro hệ thống), nghiên cứu thị trường, tiền xử lý cho các bài toán học máy phức tạp…​. Chẳng hạn, trong xử lý ảnh, PCA giúp giảm độ nhiễu và nén dữ liệu; trong học máy, PCA tiền xử lý giúp cải thiện hiệu suất phân loại, clustering.

**Nguồn tham khảo:**
[Unsupervised learning](https://www.geeksforgeeks.org/unsupervised-learning/),
[Supervised learning](https://www.geeksforgeeks.org/supervised-machine-learning/),
