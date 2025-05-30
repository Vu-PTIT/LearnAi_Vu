# I. Ensemble Learning
## A.  Định Nghĩa
-  là một kỹ thuật học máy mạnh mẽ, nơi nhiều mô hình học máy riêng lẻ (gọi là các "learner" cơ sở hoặc "weak learner") được kết hợp lại để tạo ra một mô hình dự đoán tổng hợp mạnh mẽ hơn. Nguyên tắc cơ bản là một tập thể các learner thường mang lại độ chính xác tổng thể cao hơn so với một learner đơn lẻ. Kỹ thuật này đặc biệt hữu ích khi đối mặt với các bộ dữ liệu hạn chế hoặc để cải thiện hiệu suất dự đoán của mô hình.
- Việc kết hợp nhiều mô hình trong ensemble learning mang lại nhiều lợi ích đáng kể so với việc sử dụng một mô hình đơn lẻ:
  - Giảm Overfitting (Quá Khớp): Overfitting xảy ra khi một mô hình học quá tốt trên dữ liệu huấn luyện nhưng lại hoạt động kém trên dữ liệu mới. Ensemble learning, đặc biệt là các kỹ thuật như Bagging, giúp giảm overfitting bằng cách sử dụng các tập con dữ liệu ngẫu nhiên để huấn luyện từng mô hình, tạo ra sự đa dạng và cải thiện khả năng tổng quát hóa.
  - Cải Thiện Độ Chính Xác và Hiệu Suất: Bằng cách tận dụng thế mạnh của các thuật toán đa dạng, các phương pháp ensemble nhằm mục đích giảm cả độ chệch và phương sai, dẫn đến các dự đoán đáng tin cậy hơn. Mỗi mô hình có thể vượt trội ở các khía cạnh khác nhau, chẳng hạn như nắm bắt các mẫu khác nhau hoặc xử lý các loại nhiễu cụ thể. Bằng cách kết hợp các dự đoán của chúng thông qua bỏ phiếu hoặc lấy trung bình có trọng số, các phương pháp ensemble có thể cải thiện độ chính xác tổng thể bằng cách nắm bắt sự hiểu biết toàn diện hơn về dữ liệu.
  - Tăng Cường Độ Mạnh Mẽ (Robustness): Ensemble learning tăng cường độ mạnh mẽ bằng cách xem xét ý kiến của nhiều mô hình và đưa ra dự đoán dựa trên sự đồng thuận. Điều này giảm thiểu tác động của các điểm ngoại lệ hoặc lỗi trong một mô hình đơn lẻ, đảm bảo kết quả chính xác hơn. Việc kết hợp các mô hình đa dạng làm giảm nguy cơ sai lệch hoặc không chính xác từ các mô hình riêng lẻ, nâng cao độ tin cậy và hiệu suất tổng thể của phương pháp ensemble learning.
  - Giải Quyết Vấn Đề Dữ Liệu Hạn Chế: Với những khó khăn trong việc thu thập các bộ dữ liệu lớn, được gán nhãn hợp pháp để huấn luyện các learner, ensemble learning đã có nhiều ứng dụng nhằm cải thiện hiệu suất của learner với ít dữ liệu hơn.
  - Xử Lý Dữ Liệu Chiều Cao: Các kỹ thuật ensemble learning có thể giúp giải quyết các vấn đề phát sinh từ dữ liệu chiều cao, và do đó phục vụ hiệu quả như một giải pháp thay thế cho việc giảm chiều dữ liệu
## B. Các Kỹ Thuật Ensemble Phổ Biến
### 1.Bagging (Bootstrap Aggregating)
- Bagging là kỹ thuật tổ hợp nhằm giảm phương sai (variance) và ngăn ngừa overfitting bằng cách huấn luyện nhiều mô hình con trên các tập dữ liệu con được lấy mẫu ngẫu nhiên từ tập dữ liệu gốc (có hoàn lại). Dự đoán cuối cùng được đưa ra bằng cách trung bình (đối với hồi quy) hoặc bỏ phiếu đa số (đối với phân loại) từ các mô hình con.
- Quy trình huấn luyện:
  - Lấy mẫu dữ liệu: Tạo nhiều tập dữ liệu con bằng cách lấy mẫu ngẫu nhiên có hoàn lại từ tập dữ liệu gốc.
  - Huấn luyện mô hình con: Huấn luyện một mô hình con (thường là cây quyết định) trên mỗi tập dữ liệu con.
  - Tổng hợp dự đoán: Kết hợp dự đoán từ các mô hình con bằng cách trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại).
- Đặc điểm nổi bật:
  - Giảm phương sai và overfitting.
  - Hiệu quả với dữ liệu có nhiễu.
  - Dễ dàng triển khai và song song hóa.
### 2.Random Forest(Rừng ngẫu nhiên)
- Random Forest là một biến thể của Bagging, sử dụng nhiều cây quyết định (decision trees) làm mô hình con. Mỗi cây được huấn luyện trên một tập dữ liệu con và tại mỗi nút phân chia, chỉ một tập con ngẫu nhiên của các đặc trưng được xem xét để tạo ra sự đa dạng giữa các cây, giúp giảm thiểu overfitting và cải thiện độ chính xác của mô hình.
- Quy trình huấn luyện:
  - Tạo tập dữ liệu con: Sử dụng kỹ thuật bagging để tạo nhiều tập dữ liệu con.
  - Huấn luyện cây quyết định: Huấn luyện một cây quyết định trên mỗi tập dữ liệu con, với việc chọn ngẫu nhiên một tập con của các đặc trưng tại mỗi nút phân chia.
  - Tổng hợp dự đoán: Kết hợp dự đoán từ các cây bằng cách trung bình (hồi quy) hoặc bỏ phiếu đa số (phân loại).
- Đặc điểm nổi bật:
  - Giảm phương sai và overfitting.
  - Hiệu quả với dữ liệu có nhiễu.
  - Dễ dàng triển khai và song song hóa.
### 3.AdaBoost (Adaptive Boosting)
- AdaBoost là thuật toán Boosting đầu tiên, hoạt động bằng cách huấn luyện các mô hình con tuần tự, trong đó mỗi mô hình mới cố gắng sửa lỗi của mô hình trước đó. Các mô hình con thường là các "người học yếu" (weak learners), như cây quyết định nông. AdaBoost điều chỉnh trọng số của các mẫu dữ liệu: tăng trọng số cho các mẫu bị phân loại sai để mô hình tiếp theo tập trung vào chúng.
- Quy trình huấn luyện:
  - Khởi tạo trọng số:** Gán trọng số bằng nhau cho tất cả các mẫu dữ liệu.
  - Huấn luyện mô hình con:** Huấn luyện một mô hình con trên tập dữ liệu với trọng số hiện tại.
  - Tính toán lỗi:** Tính toán lỗi của mô hình con trên tập dữ liệu.
  - Tính toán trọng số mô hình:** Tính toán trọng số của mô hình con dựa trên lỗi.
  - Cập nhật trọng số mẫu:** Tăng trọng số cho các mẫu bị phân loại sai và giảm trọng số cho các mẫu được phân loại đúng.
  - Lặp lại: Lặp lại các bước trên cho đến khi đạt được số lượng mô hình con mong muốn hoặc lỗi giảm xuống dưới ngưỡng cho phép.
- Đặc điểm nổi bật:
  - Giảm độ chệch và cải thiện độ chính xác.
  - Hiệu quả với dữ liệu không cân bằng.
  - Dễ bị overfitting nếu không điều chỉnh đúng.
### 4.XGBoost (Extreme Gradient Boosting)
- XGBoost là một phiên bản nâng cao của Boosting, sử dụng kỹ thuật tối ưu hóa gradient và regularization để cải thiện hiệu suất và ngăn ngừa overfitting. XGBoost nổi tiếng với hiệu suất cao và khả năng xử lý dữ liệu lớn.
- Quy trình huấn luyện:
  - Khởi tạo mô hình: Bắt đầu với một mô hình đơn giản.
  - Tính toán gradient: Tính toán gradient của hàm mất mát đối với dự đoán hiện tại.
  - Huấn luyện mô hình con: Huấn luyện một mô hình con để dự đoán gradient.
  - Cập nhật mô hình: Cập nhật mô hình bằng cách cộng mô hình con mới với một hệ số học.
  - Lặp lại: Lặp lại các bước trên cho đến khi đạt được số lượng mô hình con mong muốn hoặc lỗi giảm xuống dưới ngưỡng cho phép.
- Đặc điểm nổi bật:
  - Giảm độ chệch và cải thiện độ chính xác.
  - Hiệu quả với dữ liệu không cân bằng.
  - Có khả năng xử lý dữ liệu lớn và hỗ trợ regularization.
### 5. Stacking (Stacked Generalization)
- Stacking là kỹ thuật tổ hợp kết hợp nhiều mô hình con (có thể khác nhau về loại) bằng cách sử dụng một mô hình meta (meta-model) để học cách kết hợp dự đoán từ các mô hình con.
- Quy trình huấn luyện:
  - Tầng cơ sở (base level): Huấn luyện các mô hình con trên tập dữ liệu gốc.
  - Tạo tập dữ liệu meta: Sử dụng dự đoán của các mô hình con trên tập validation để tạo tập dữ liệu meta.
  - Tầng meta (meta level): Huấn luyện mô hình meta trên tập dữ liệu meta.
  - Dự đoán: Để dự đoán cho dữ liệu mới, sử dụng các mô hình con để tạo đặc trưng và đưa vào mô hình meta để dự đoán cuối cùng.
- Đặc điểm nổi bật:
  - Khai thác sức mạnh của nhiều mô hình khác nhau.
  - Thường cho hiệu suất cao hơn so với các mô hình đơn lẻ.
  - Phức tạp trong việc triển khai và điều chỉnh.
### 6.Voting Classifier
- Voting Classifier là một kỹ thuật ensemble học máy kết hợp nhiều mô hình riêng lẻ để đưa ra dự đoán dựa trên quyết định đa số hoặc xác suất trung bình của các mô hình đó. Ý tưởng là thay vì tạo các mô hình chuyên dụng riêng biệt và tìm độ chính xác cho từng mô hình, chúng ta tạo một mô hình duy nhất huấn luyện bằng các mô hình này và dự đoán đầu ra dựa trên đa số phiếu kết hợp của chúng cho mỗi lớp đầu ra. Có hai loại chính:
  - Hard Voting: Dự đoán cuối cùng là lớp được nhiều mô hình con dự đoán nhất.
  - Soft Voting: Dự đoán cuối cùng dựa trên trung bình xác suất dự đoán từ các mô hình con.
- Quy trình huấn luyện:
  - Huấn luyện mô hình con: Huấn luyện các mô hình con trên tập dữ liệu gốc.
  - Dự đoán: Dự đoán cho dữ liệu mới bằng cách kết hợp kết quả từ các mô hình con bằng hard voting hoặc soft voting.
- Đặc điểm nổi bật:
  - Dễ triển khai và hiểu.
  - Cải thiện độ chính xác bằng cách kết hợp nhiều mô hình.
  - Hiệu suất phụ thuộc vào chất lượng và sự đa dạng của các mô hình con.
# II. Semi-Supervised Learning (Học bán giám sát)
## A. Khái niệm
- Học Bán Giám Sát (SSL) là một nhánh của học máy nằm giữa học có giám sát (sử dụng hoàn toàn dữ liệu đã gán nhãn) và học không giám sát (sử dụng hoàn toàn dữ liệu chưa gán nhãn). Kỹ thuật này tận dụng một lượng nhỏ dữ liệu đã gán nhãn cùng với một lượng lớn dữ liệu chưa gán nhãn để huấn luyện mô hình.
- SSL đặc biệt hữu ích trong các tình huống sau:
  - Phân loại hình ảnh y tế: Gán nhãn cho hình ảnh y tế đòi hỏi chuyên môn cao và tốn kém thời gian. SSL giúp tận dụng dữ liệu không nhãn để cải thiện mô hình chẩn đoán.
  - Xử lý ngôn ngữ tự nhiên: Trong các bài toán như phân loại văn bản hoặc phân tích cảm xúc, việc gán nhãn dữ liệu có thể tốn kém. SSL cho phép sử dụng dữ liệu văn bản không nhãn để nâng cao hiệu suất mô hình.
  - Phát hiện gian lận: Trong lĩnh vực tài chính, dữ liệu về các giao dịch gian lận thường rất ít. SSL giúp mô hình học từ dữ liệu không nhãn để phát hiện các mẫu gian lận mới.
## B. Các phương pháp cơ bản
### 1. Pseudo-Labeling
- Phương pháp này gán nhãn giả cho dữ liệu không nhãn dựa trên dự đoán của mô hình hiện tại.
Sau đó, dữ liệu có nhãn giả được sử dụng cùng với dữ liệu có nhãn thật để huấn luyện mô hình.
Quá trình này có thể được lặp lại nhiều lần để cải thiện độ chính xác của mô hình.
### 2. Self-Training
- Đây là một trong những phương pháp SSL đơn giản và phổ biến nhất.
- Quy Trình Hoạt Động :
Huấn luyện một mô hình ban đầu (base model) chỉ sử dụng một lượng nhỏ dữ liệu đã gán nhãn.
Sử dụng mô hình đã huấn luyện này để dự đoán nhãn cho dữ liệu chưa gán nhãn. Những nhãn được dự đoán này được gọi là "nhãn giả" (pseudo-labels).
Chọn ra các dự đoán có độ tin cậy cao nhất (ví dụ: xác suất dự đoán vượt một ngưỡng nhất định).
Thêm các mẫu dữ liệu chưa gán nhãn này cùng với nhãn giả tự tin của chúng vào tập dữ liệu huấn luyện đã gán nhãn ban đầu.
Huấn luyện lại mô hình trên tập dữ liệu huấn luyện mở rộng này.
Lặp lại các bước 2-5 cho đến khi không còn dữ liệu chưa gán nhãn hoặc đạt được một tiêu chí dừng nào đó (ví dụ: số vòng lặp tối đa, không có thêm nhãn giả tự tin nào được tạo ra).
### 3. Co-Training
- Co-training là một phiên bản cải tiến của self-training, đặc biệt hữu ích khi dữ liệu có thể được mô tả bằng hai "khung nhìn" (views) đặc trưng khác nhau, độc lập có điều kiện với nhau dựa trên lớp.42 Mỗi khung nhìn phải đủ để phân loại dữ liệu một mình.
- Quy Trình Hoạt Động :
  - Huấn luyện hai bộ phân loại riêng biệt (classifier 1 và classifier 2), mỗi bộ trên một khung nhìn đặc trưng khác nhau của dữ liệu đã gán nhãn.
  - Mỗi bộ phân loại sau đó dự đoán nhãn cho dữ liệu chưa gán nhãn dựa trên khung nhìn tương ứng của nó.
  - Các dự đoán có độ tin cậy cao nhất từ classifier 1 (dựa trên khung nhìn 1) được sử dụng làm dữ liệu huấn luyện bổ sung cho classifier 2 (được huấn luyện trên khung nhìn 2), và ngược lại. Tức là, nếu classifier 1 tự tin dự đoán nhãn chính xác cho một mẫu dữ liệu trong khi classifier kia mắc lỗi dự đoán, thì dữ liệu với nhãn giả tự tin do classifier 1 gán sẽ cập nhật classifier 2 và ngược lại.
  - Quá trình này được lặp lại, cho phép hai bộ phân loại "dạy" lẫn nhau.
  - Dự đoán cuối cùng có thể được kết hợp từ hai bộ phân loại đã được cập nhật.
### 4. Graph-Based Label Propagation
- Các phương pháp này biểu diễn cả dữ liệu có nhãn và không có nhãn dưới dạng các nút trong một đồ thị, với các cạnh biểu thị sự tương đồng hoặc mối quan hệ giữa các điểm dữ liệu. Thông tin nhãn sau đó được "lan truyền" (propagated) từ các nút đã gán nhãn sang các nút chưa gán nhãn thông qua cấu trúc đồ thị.
- Quy Trình Hoạt Động (ví dụ: Label Propagation) :
  - Xây Dựng Đồ Thị: Tạo một đồ thị trong đó mỗi điểm dữ liệu (cả có nhãn và không có nhãn) là một nút. Các cạnh được hình thành giữa các nút dựa trên sự gần gũi hoặc tương đồng của chúng (ví dụ: sử dụng đồ thị k-láng giềng gần nhất - k-NN graph). Trọng số của các cạnh có thể biểu thị mức độ tương đồng.
  - Lan Truyền Nhãn: Các nút đã gán nhãn ban đầu được "tô màu" với nhãn của chúng.
  - Thuật toán lặp đi lặp lại việc lan truyền thông tin nhãn qua các cạnh của đồ thị. Một nút chưa gán nhãn sẽ nhận nhãn dựa trên nhãn của các nút lân cận của nó, thường là theo đa số hoặc theo trọng số của các cạnh. Ví dụ, đối với một nút chưa được gán nhãn, thuật toán đếm tất cả các đường đi khác nhau qua mạng từ nút chưa được gán nhãn đó đến từng nút đã được tô màu (đã gán nhãn). Dựa trên số lượng đường đi, nút chưa được gán nhãn được gán nhãn của lớp mà nó có nhiều kết nối hơn.
  - Quá trình này tiếp tục cho đến khi nhãn của các nút hội tụ (không thay đổi nữa) hoặc đạt đến một số lần lặp nhất định.

# III. Probabilistic Graphical Models (PGM)
## A. Khái niệm cơ bản
- Mô Hình Đồ Thị Xác Suất (PGM) là một khuôn khổ phong phú để mã hóa các phân phối xác suất trên các miền phức tạp, đặc biệt là các phân phối xác suất đồng thời (joint distributions) trên một số lượng lớn các biến ngẫu nhiên tương tác với nhau. Các biểu diễn này nằm ở giao điểm của thống kê và khoa học máy tính, dựa trên các khái niệm từ lý thuyết xác suất, thuật toán đồ thị, học máy, v.v..PGM kết hợp sức mạnh của lý thuyết xác suất với lý thuyết đồ thị để tạo ra một khuôn khổ linh hoạt để biểu diễn các phụ thuộc phức tạp giữa các biến ngẫu nhiên
### 1. Bayesian Networks (Mạng Bayes)
- Mạng Bayes (BN), còn được gọi là mạng niềm tin (belief networks) hoặc mô hình đồ thị có hướng, sử dụng đồ thị có hướng không chu trình (Directed Acyclic Graph - DAG) để biểu diễn các mối quan hệ xác suất và phụ thuộc có điều kiện giữa một tập hợp các biến.
- Mô hình đồ thị có hướng (Directed Acyclic Graph - DAG).
- Các nút biểu diễn các biến ngẫu nhiên.
- Các cung thể hiện mối quan hệ nhân quả có điều kiện.
- Mỗi nút gắn với một bảng phân phối xác suất có điều kiện (CPT - Conditional Probability Table).

### 2. Markov Networks (Markov Random Fields)
- Mạng Markov (MN), còn được gọi là Trường Ngẫu Nhiên Markov (Markov Random Fields - MRF) hoặc mô hình đồ thị vô hướng, sử dụng đồ thị vô hướng để biểu diễn các phụ thuộc xác suất giữa một tập hợp các biến.
- Mô hình đồ thị vô hướng (Undirected Graph).
- Biểu diễn sự phụ thuộc giữa các biến qua các "clique" (nhóm biến liên thông).
- Mỗi clique gắn với một hàm thế (potential function) dùng để tính xác suất toàn cục.
## B. Ứng dụng của PGM trong Machine Learning
- Ứng Dụng BN :
  - Chẩn Đoán Y Khoa: Mô hình hóa mối quan hệ phức tạp giữa triệu chứng, bệnh tật, tiền sử bệnh nhân và các yếu tố nguy cơ để hỗ trợ chẩn đoán. Ví dụ, xác định xác suất một bệnh nhân mắc bệnh tim dựa trên tuổi, giới tính, mức cholesterol và thói quen hút thuốc.
  - Xử Lý Ngôn Ngữ Tự Nhiên (NLP): Phân loại văn bản, phân tích tình cảm, mô hình hóa chủ đề. Ví dụ, xác định tình cảm của một văn bản (tích cực, tiêu cực hoặc trung tính) dựa trên các từ được sử dụng.
  - Dự Báo Thời Tiết: Mô hình hóa mối quan hệ giữa các biến như nhiệt độ, độ ẩm, tốc độ gió để dự đoán điều kiện thời tiết trong tương lai.
  - Khám Phá Tri Thức và Suy Luận Nhân Quả từ Dữ Liệu Lớn.
  - Mạng Nơ-ron Bayes (BNN): Một sự mở rộng kết hợp sự không chắc chắn vào các mạng nơ-ron truyền thống.
  - Phân Tích Rủi Ro Kinh Tế, Tin Sinh Học, Xử Lý Hình Ảnh.
- Ứng Dụng MN:
  - Thị Giác Máy Tính và Xử Lý Hình Ảnh: Phân đoạn ảnh (image segmentation), khử nhiễu ảnh (image denoising), tạo kết cấu (texture generation), khôi phục ảnh mờ (dehazing). Ví dụ, trong bài toán khử sương mù, thế năng clique có thể được mã hóa trên cường độ pixel theo cặp, áp đặt ràng buộc theo ngữ cảnh về sự thay đổi cường độ của pixel lân cận.
  - Xử Lý Ngôn Ngữ Tự Nhiên: Mô hình hóa các mối quan hệ giữa các từ hoặc thực thể trong văn bản.
  - Tin Sinh Học Tính Toán: Phân tích mạng tương tác gen hoặc protein.
  - Mạng Xã Hội: Mô hình hóa ảnh hưởng và tương tác giữa các cá nhân.
  - Tối Ưu Hóa Tổ Hợp. Một biến thể đáng chú ý là Trường Ngẫu Nhiên Có Điều Kiện (Conditional Random Fields - CRF), trong đó mỗi biến ngẫu nhiên cũng có thể được điều kiện hóa trên một tập hợp các quan sát toàn cục.
# IV. Recommendation Systems
## A. Các phương pháp cơ bản
- Hệ thống gợi ý (Recommendation Systems) là các công cụ và kỹ thuật cung cấp đề xuất các "mục" (items) cho người dùng. Các mục này có thể là sản phẩm để mua, bài hát để nghe, phim để xem, hoặc bất kỳ thứ gì khác mà người dùng có thể quan tâm. Mục tiêu chính là giúp người dùng khám phá các mục mà họ có thể thích từ một danh mục lớn, qua đó nâng cao trải nghiệm người dùng, tăng sự tương tác và thường là thúc đẩy doanh số bán hàng hoặc tiêu thụ nội dung.
### 1. Content-Based Filtering
- Lọc dựa trên nội dung đề xuất các mục tương tự như những mục mà người dùng đã thích trong quá khứ, dựa trên đặc điểm hoặc nội dung của các mục đó và hồ sơ của người dùng.
- Cơ Chế Hoạt Động :
  - Lập Hồ Sơ Mục (Item Profiling): Mỗi mục được biểu diễn bằng các đặc trưng (metadata) của nó. Ví dụ, đối với phim, đó có thể là thể loại (hài, kinh dị, lãng mạn), diễn viên, đạo diễn, từ khóa. Đối với sản phẩm, đó có thể là danh mục, mô tả, thông số kỹ thuật.
  - Lập Hồ Sơ Người Dùng (User Profiling): Một hồ sơ về sở thích của người dùng được xây dựng dựa trên các tương tác trong quá khứ của họ (ví dụ: các mục đã xếp hạng, xem, mua) và các đặc trưng của những mục đó. Ví dụ, nếu một người dùng xem nhiều phim "hành động", hồ sơ của họ sẽ cho thấy sở thích đối với "hành động".
  - Đề Xuất: Hệ thống so khớp hồ sơ của người dùng với hồ sơ của các mục mới và đề xuất các mục có độ tương tự cao (ví dụ: sử dụng thuật toán k-Nearest Neighbors (k-NN) trên các vector đặc trưng ).
### 2. Collaborative Filtering
- Lọc cộng tác (CF) đề xuất các mục dựa trên nguyên tắc rằng những người dùng đã đồng ý trong quá khứ có xu hướng đồng ý trong tương lai. Nó tận dụng hành vi tập thể của người dùng ("trí tuệ đám đông").
- Nguyên Tắc Cốt Lõi: "Những người dùng tương tự thích những mục tương tự." Nó nhóm người dùng dựa trên hành vi tương tự và đề xuất các mục được nhóm đó thích.
Ma Trận Tương Tác Người Dùng - Mục: Thường sử dụng một ma trận các xếp hạng hoặc tương tác của người dùng (ví dụ: mua hàng, lượt xem) đối với các mục.
- Các Cách Tiếp Cận:
  - CF Dựa trên Người Dùng (User-Based CF): Tìm những người dùng tương tự với người dùng mục tiêu (hàng xóm) dựa trên lịch sử tương tác của họ. Đề xuất các mục được những người dùng tương tự này thích nhưng người dùng mục tiêu chưa xem.61 Độ tương tự được tính giữa các hàng của ma trận người dùng-mục.
  - CF Dựa trên Mục (Item-Based CF): Đề xuất các mục tương tự như những mục mà người dùng mục tiêu đã thích trong quá khứ. Độ tương tự của mục được xác định bởi cách người dùng tương tác với chúng (ví dụ: các mục thường được mua cùng nhau hoặc xếp hạng cùng nhau bởi nhiều người dùng được coi là tương tự). Độ tương tự được tính giữa các cột của ma trận người dùng-mục. Amazon là người tiên phong trong CF item-to-item.
  - CF Dựa trên Mô Hình (Model-Based CF) (ví dụ: Phân Tích Nhân Tử Ma Trận - Matrix Factorization): Học các yếu tố tiềm ẩn (latent factors) cho người dùng và mục từ ma trận tương tác để dự đoán các xếp hạng còn thiếu. Các kỹ thuật bao gồm Phân tích giá trị suy biến (Singular Value Decomposition - SVD), Bình phương tối thiểu xen kẽ (Alternating Least Squares - ALS).
## 2. Ứng dụng thực tế
- Netflix:
  - Kết hợp Collaborative Filtering và Content-Based để đề xuất phim dựa trên lịch sử xem và đánh giá của người dùng.
  - Sử dụng Deep Learning để phân tích metadata (tên phim, thể loại, diễn viên) và hành vi người dùng.
- Amazon:
  - Gợi ý sản phẩm dựa trên lịch sử mua sắm, sản phẩm đã xem, giỏ hàng, và sản phẩm tương tự đã mua bởi người dùng khác.
  - Dùng cả Collaborative Filtering (ví dụ: "người dùng mua A cũng mua B") và Content-Based Filtering (ví dụ: sản phẩm tương tự).
- Spotify:
  - Sử dụng mô hình học sâu để gợi ý bài hát dựa trên sở thích nghe nhạc, hành vi tương tác, cũng như đặc điểm âm thanh (audio features).
  - Hệ thống gợi ý "Discover Weekly" dựa vào collaborative filtering và phân tích nội dung bài hát.
# V. Giới thiệu MLOps
## A. Định Nghĩa
- MLOps (Machine Learning Operations) là một tập hợp các thực tiễn, một văn hóa và một kỷ luật nhằm mục đích thống nhất việc phát triển ứng dụng ML (Dev) với việc triển khai và vận hành hệ thống ML (Ops). Nó tập trung vào việc hợp lý hóa và tự động hóa toàn bộ vòng đời ML, từ thu thập dữ liệu và phát triển mô hình đến triển khai, giám sát và bảo trì.
- MLOps rất quan trọng để quản lý một cách có hệ thống việc phát hành các mô hình ML mới, tương tự như cách DevOps quản lý tài sản phần mềm. Nếu không có MLOps, các tổ chức phải đối mặt với những thách thức như gia tăng lỗi, thiếu khả năng mở rộng, giảm hiệu quả và hợp tác kém. Nó giải quyết sự phức tạp của việc vận hành các mô hình ML, đảm bảo chúng mạnh mẽ, có thể mở rộng và hiệu quả trong sản xuất. MLOps giúp triển khai các cải tiến ML một cách nhất quán và đáng tin cậy.
## B. ML Pipeline là gì?
- Một quy trình ML (ML pipeline) là một chuỗi các bước có cấu trúc nhằm tự động hóa, chuẩn hóa và đơn giản hóa quy trình làm việc liên quan đến việc xây dựng, đào tạo, đánh giá và triển khai các mô hình học máy.Thay vì tập trung vào việc triển khai một mô hình duy nhất, các quy trình ML nhằm mục đích xây dựng các hệ thống hỗ trợ phát triển, thử nghiệm và triển khai liên tục thông qua tự động hóa. Điều này rất quan trọng vì xu hướng dữ liệu thay đổi và các mô hình cần được đào tạo lại thường xuyên để luôn cập nhật và cung cấp các dự đoán chất lượng cao
- Các Loại Quy Trình Phổ Biến :
  - Quy Trình Theo Lô (Batch Pipelines): Xử lý dữ liệu theo các khoảng thời gian đã định (ví dụ: hàng ngày, hàng tuần). Hữu ích cho việc đào tạo mô hình trên các bộ dữ liệu lớn.
  - Quy Trình Thời Gian Thực (Real-Time Pipelines): Xử lý dữ liệu ngay khi nó được tạo ra. Lý tưởng cho các ứng dụng yêu cầu phản hồi ngay lập tức như phát hiện gian lận, công cụ gợi ý hoặc định giá động.

## C. Các công cụ phổ biến trong MLOps
- Databricks :
  - Đặc Điểm: Mặc dù không được mô tả chi tiết như một nền tảng MLOps độc lập trong các đoạn trích được cung cấp, Azure ML có các tích hợp nổi bật với nền tảng Databricks.82 Databricks thường được biết đến với khả năng xử lý dữ liệu lớn và học máy dựa trên Spark. MLflow (xem bên dưới) được tạo bởi Databricks.
- Kubeflow :
  - Đặc Điểm: Bộ công cụ MLOps mã nguồn mở, gốc Kubernetes để xây dựng và quản lý các quy trình ML di động, có thể kết hợp. Cung cấp sự linh hoạt để điều phối đào tạo, tinh chỉnh và phục vụ bằng các khái niệm trừu tượng Kubernetes quen thuộc. Yêu cầu kiến thức sâu về cơ sở hạ tầng.
  - Trường Hợp Sử Dụng Tốt Nhất: Các nhóm có chuyên môn Kubernetes vững chắc muốn tùy chỉnh và kiểm soát hoàn toàn các quy trình MLOps của họ, đặc biệt trong các môi trường được quản lý chặt chẽ hoặc đám mây lai.
- MLflow :
  - Đặc Điểm: Nền tảng MLOps mã nguồn mở, nhẹ, được tạo bởi Databricks, tập trung vào việc quản lý thử nghiệm ML và quản lý phiên bản mô hình. Các thành phần mô-đun cho phép các nhóm tích hợp theo dõi, đăng ký và triển khai vào các quy trình hiện có của họ.
  - Trường Hợp Sử Dụng Tốt Nhất: Các nhóm ML tìm kiếm công cụ nhẹ, có thể tùy chỉnh để theo dõi thử nghiệm, chia sẻ mô hình và quản lý phiên bản mà không phụ thuộc vào một nền tảng quy mô lớn hoặc Kubernetes.
## D. Các bước chính trong pipeline MLOps
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