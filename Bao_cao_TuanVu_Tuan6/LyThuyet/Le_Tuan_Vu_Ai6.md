# I.  Giới Thiệu về Mạng Nơ-ron (Neural Networks)
## A. Mạng Nơ-ron là gì?
- Mạng nơ-ron (Neural Networks - NNs) là một loại thuật toán học máy được lấy cảm hứng từ cấu trúc và chức năng của bộ não con người.1 Chúng là công cụ mạnh mẽ, vượt trội trong việc giải quyết các vấn đề phức tạp mà các thuật toán máy tính truyền thống khó xử lý, chẳng hạn như nhận dạng hình ảnh và xử lý ngôn ngữ tự nhiên. 
- Mạng nơ-ron bao gồm:
  -  Các nút (nodes) được kết nối với nhau, gọi là các nơ-ron (neurons), thường được sắp xếp thành các lớp (layers). Mỗi nơ-ron nhận đầu vào từ các nơ-ron khác, xử lý thông tin đó và truyền một đầu ra đến các nơ-ron khác. 
  -  Các kết nối giữa các nơ-ron có các trọng số (weights) liên kết, biểu thị sức mạnh của kết nối đó. Trong quá trình huấn luyện, mạng sẽ điều chỉnh các trọng số này để tinh chỉnh hiệu suất của nó đối với một nhiệm vụ nhất định. Quá trình học này cho phép chúng đưa ra dự đoán và nhận dạng các mẫu, thúc đẩy việc áp dụng rộng rãi của chúng trong các ứng dụng đa dạng.
- Cấu trúc phân lớp không chỉ là một lựa chọn tổ chức mà còn là yếu tố cơ bản quyết định cách mạng nơ-ron học các biểu diễn phân cấp của dữ liệu. Các đặc trưng đơn giản hơn được học ở các lớp đầu, sau đó được kết hợp để tạo thành các đặc trưng phức tạp hơn ở các lớp sâu hơn. Điều này đặc biệt rõ ràng trong các mạng học sâu
(deep learning networks)

## B. Tại sao cần dùng Mạng Nơ-ron? Ưu điểm so với các mô hình Học Máy truyền thống
- Mạng nơ-ron đã chứng tỏ sự thành thạo trong việc giải quyết các vấn đề phức tạp mà các thuật toán truyền thống gặp nhiều thách thức, đặc biệt là trong các lĩnh vực như nhận dạng hình ảnh và Xử lý Ngôn ngữ Tự nhiên (NLP). Sự vượt trội này xuất phát từ một số ưu điểm chính:
  - Khả năng thích ứng (Adaptability): Mạng nơ-ron có khả năng học và thích ứng
với dữ liệu mới, làm cho chúng linh hoạt và mạnh mẽ hơn so với các thuật toán
truyền thống.1 Khi có dữ liệu mới, mạng có thể được huấn luyện lại hoặc tinh chỉnh
để cải thiện hiệu suất mà không cần thiết kế lại từ đầu.
  - Phi tuyến tính (Non-Linearity): Chúng có thể mô hình hóa các mối quan hệ phi
tuyến phức tạp trong dữ liệu, một hạn chế phổ biến của nhiều mô hình học máy
truyền thống.1 Hầu hết dữ liệu trong thế giới thực đều có tính chất phi tuyến, và
khả năng nắm bắt các mối quan hệ này là một lợi thế đáng kể. Nếu không có các
hàm kích hoạt phi tuyến, ngay cả các mạng sâu cũng sẽ bị giới hạn trong việc giải
quyết các vấn đề phân tách tuyến tính đơn giản.
  - Tự động hóa & Hiệu quả (Automation & Efficiency): Mạng nơ-ron có thể tự
động hóa các tác vụ trước đây do con người thực hiện, giúp tiết kiệm thời gian và
tài nguyên, đồng thời cải thiện các quy trình kinh doanh.1
  - Cải thiện việc ra quyết định (Improved Decision-Making): Mạng nơ-ron có thể
cung cấp những hiểu biết sâu sắc mà khó có thể thu được bằng các phương pháp
truyền thống, từ đó hỗ trợ việc ra quyết định tốt hơn.1
  - Khả năng chịu lỗi (Fault Tolerance): Mạng nơ-ron có thể chịu được nhiễu và dữ
liệu bị thiếu ở một mức độ nhất định, làm cho chúng đáng tin cậy hơn trong một
số trường hợp

## C. Perceptron Đa Lớp (Multilayer Perceptron - MLP)
- Perceptron Đa Lớp (MLP) là một loại mạng nơ-ron nhân tạo truyền thẳng (feedforward) cơ bản và là một trong những kiến trúc nền tảng trong lĩnh vực học sâu. Nó bao gồm nhiều lớp nơ-ron được sắp xếp theo cấu trúc phân cấp.

### 1. Kiến trúc cơ bản: Lớp Đầu vào, Lớp Ẩn, và Lớp Đầu ra
- Một MLP bao gồm ít nhất ba loại lớp nơ-ron, và các lớp này được kết nối đầy đủ (fully connected), nghĩa là mỗi nút trong một lớp được kết nối với mọi nút trong lớp tiếp theo.
  - Lớp Đầu vào (Input Layer): Lớp này nhận dữ liệu hoặc đặc trưng ban đầu. Số lượng nơ-ron trong lớp đầu vào thường tương ứng với số lượng đặc trưng đầu vào của dữ liệu.5 Ví dụ, nếu dữ liệu đầu vào là một hình ảnh có kích thước 28x28 pixel, lớp đầu vào có thể có 28×28=784 nơ-ron.
  - Lớp Ẩn (Hidden Layers): Một hoặc nhiều lớp nơ-ron nằm giữa lớp đầu vào và lớp đầu ra. Các lớp này thực hiện các phép tính toán và biến đổi dữ liệu đầu vào, cho phép mạng học các mẫu phức tạp.5 Chính sự tồn tại của các lớp ẩn này, đặc biệt là nhiều lớp ẩn, tạo nên khái niệm "sâu" (deep) trong học sâu. Các kích hoạt (outputs) của các nơ-ron trong lớp ẩn không được diễn giải trực tiếp dưới dạng đầu vào hay đầu ra cuối cùng, mà chúng đại diện cho các đặc trưng trừu tượng mà mạng đã học được. Sức mạnh của MLP (và mạng nơ-ron nói chung) đến từ khả năng học các biểu diễn trung gian, trừu tượng này. Chẳng hạn, trong một tác vụ nhận dạng hình ảnh, các lớp ẩn ban đầu có thể học cách phát hiện các cạnh hoặc góc, trong khi các lớp ẩn sâu hơn có thể học cách nhận diện các bộ phận phức tạp hơn như mắt hoặc mũi, và cuối cùng là toàn bộ đối tượng
  - Lớp Đầu ra (Output Layer): Lớp này tạo ra kết quả cuối cùng của mạng, ví dụ như điểm số phân loại cho mỗi lớp trong bài toán phân loại, hoặc một giá trị dự đoán trong bài toán hồi quy.5 Số lượng nơ-ron trong lớp đầu ra phụ thuộc vào nhiệm vụ cụ thể. Ví dụ, trong bài toán phân loại 10 lớp, lớp đầu ra sẽ có 10 nơ-ron.

### 2. Trọng số và Độ lệch: Các Tham số có thể học được.
- Các tham số có thể học được của mạng nơ-ron, được điều chỉnh trong quá trình huấn
luyện, là trọng số và độ lệch.
  - Trọng số (Weights): Trọng số là các tham số xác định độ mạnh của kết nối giữa các nơ-ron ở các lớp liền kề.6 Chúng nhân rộng tín hiệu đầu vào từ một nơ-ron này sang nơ-ron tiếp theo.8 Về cơ bản, trọng số quy định mức độ ảnh hưởng của một đầu vào nhất định đối với đầu ra của nơ-ron.
  - Độ lệch (Biases): Độ lệch là các tham số bổ sung được cộng vào tổng có trọng số của các đầu vào trước khi áp dụng hàm kích hoạt.6 Chúng cho phép dịch chuyển đầu ra của hàm kích hoạt, cung cấp sự linh hoạt để phù hợp với dữ liệu không đi qua gốc tọa độ.8 Mỗi nơ-ron trong một lớp (trừ lớp đầu vào) thường có độ lệch riêng.8 Độ lệch rất quan trọng đối với tính linh hoạt của mô hình. Nếu không có độ lệch, kích hoạt của một nơ-ron sẽ chỉ là tổng trọng số của các đầu vào của nó. Nếu tất cả các đầu vào bằng không, đầu ra (trước khi kích hoạt) cũng sẽ bằng không. Độ lệch cho phép các nơ-ron kích hoạt ngay cả khi tất cả đầu vào bằng không, cho phép mô hình học các độ dời (offsets) và phù hợp với một phạm vi hàm rộng hơn.

### 3. Hàm Kích hoạt: Giới thiệu tính Phi tuyến.
- Hàm kích hoạt đóng một vai trò cơ bản trong mạng nơ-ron bằng cách đưa tính phi tuyến vào đầu ra của một nơ-ron. Tính phi tuyến này là cực kỳ quan trọng vì nếu không có nó, một mạng đa lớp sẽ hoạt động giống như một mô hình tuyến tính đơn lẻ, bất kể có bao nhiêu lớp. Điều này sẽ hạn chế nghiêm trọng khả năng của mạng trong việc học các mẫu phức tạp thường thấy trong hầu hết dữ liệu thực tế.

### 4. Quá trình Truyền thẳng (Forward Pass): Dữ liệu lan truyền qua MLP như thế nào.
- Quá trình truyền thẳng (forward pass hay forward propagation) là quá trình dữ liệu
đầu vào được đưa qua mạng, từ lớp này sang lớp khác, để tạo ra một đầu ra.5 Tại mỗi
nơ-ron trong một lớp (trừ lớp đầu vào):
  - Tính tổng có trọng số của các đầu vào từ lớp trước đó. Mỗi đầu vào được nhân với
trọng số tương ứng của kết nối.
  - Cộng thêm giá trị độ lệch (bias) của nơ-ron đó.
  - Đưa kết quả tổng này qua hàm kích hoạt của nơ-ron. Đầu ra của hàm kích hoạt
này sau đó trở thành đầu vào cho các nơ-ron ở lớp tiếp theo. Quá trình này lặp lại
cho đến khi đến lớp đầu ra, nơi mạng tạo ra dự đoán cuối cùng

### 5. Quá trình Truyền ngược (Backward Pass): Học từ Lỗi
- Sau khi quá trình truyền thẳng hoàn tất và mạng đưa ra một dự đoán, dự đoán này được so sánh với giá trị mục tiêu thực tế bằng cách sử dụng một hàm mất mát (loss function) để định lượng sai số.5 Quá trình truyền ngược, hay còn gọi là lan truyềnngược (backpropagation), sau đó tính toán gradient (đạo hàm riêng) của hàm mất mát
đối với từng trọng số và độ lệch trong mạng.
- Điều này được thực hiện bằng cách lan truyền lỗi ngược lại qua mạng, từ lớp đầu ra đến lớp đầu vào, sử dụng quy tắc chuỗi (chain rule) của giải tích.5 Các gradient này cho biết mỗi tham số đã đóng góp vào tổng lỗi như thế nào và theo hướng nào cần điều chỉnh chúng để giảm thiểu lỗi.
- Quá trình truyền ngược chính là cơ chế cho phép "học" theo nghĩa quy trách nhiệm lỗi. Bằng cách tính toán mức độ mà mỗi trọng số/độ lệch đóng góp vào lỗi tổng thể, mạng biết cách điều chỉnh chúng để cải thiện. Hiệu quả của thuật toán lan truyền ngược (tính toán tất cả các gradient với độ phức tạp tính toán gần bằng một lượt truyền thẳng) là một bước đột phá giúp cho việc huấn luyện các mạng sâu trở nên khả thi. Nếu không có nó, việc tính toán gradient sẽ cực kỳ tốn kém.

## D. Huấn luyện Mạng Nơ-ron
- Huấn luyện mạng nơ-ron là một quá trình lặp đi lặp lại nhằm điều chỉnh các trọng số
và độ lệch của mạng để giảm thiểu sự khác biệt giữa các đầu ra dự đoán và các đầu ra
thực tế (nhãn). Quá trình này dựa trên việc tối ưu hóa một hàm mất mát.
### 1. Vai trò của Hàm Mất mát (ví dụ: MSE, Cross-Entropy).
Hàm mất mát (loss function hay cost function) là một thước đo định lượng mức độ sai
lệch giữa dự đoán của mô hình và giá trị thực tế; nó định lượng lỗi của mô hình.7 Mục
tiêu của quá trình huấn luyện là tìm ra bộ tham số (trọng số và độ lệch) sao cho hàm
mất mát này đạt giá trị nhỏ nhất.
  - Sai số Toàn phương Trung bình (Mean Squared Error - MSE): Thường được sử
dụng cho các tác vụ hồi quy (regression), nơi đầu ra là một giá trị liên tục. MSE
tính trung bình của bình phương sự khác biệt giữa giá trị dự đoán và giá trị thực
tế.14 Việc bình phương đảm bảo rằng các lỗi luôn dương và phạt nặng hơn đối với
các lỗi lớn.14 Công thức của nó là MSE=n1∑i=1n(yi−y^i)2, trong đó n là số lượng
mẫu, yi là giá trị thực và y^i là giá trị dự đoán.
  - Mất mát Entropy Chéo (Cross-Entropy Loss): Thường được sử dụng cho các
tác vụ phân loại (classification). Nó đo lường sự khác biệt giữa phân phối xác suất
dự đoán và phân phối xác suất thực tế của các lớp.13 Đối với phân loại nhị phân,
công thức có thể là −(ylog(p)+(1−y)log(1−p)), trong đó y là nhãn thực (0 hoặc 1) và
p là xác suất dự đoán.

### 2. Gradient Descent: Tối ưu hóa Tham số Mạng.
  - Gradient Descent (GD) là một thuật toán tối ưu hóa lặp đi lặp lại được sử dụng để tìm
giá trị cực tiểu của hàm mất mát bằng cách điều chỉnh các tham số mạng (trọng số và
độ lệch).6 Ý tưởng cốt lõi là các tham số được cập nhật theo hướng ngược lại với
gradient của hàm mất mát đối với các tham số đó.12 Điều này có nghĩa là nếu gradient
dương, tham số sẽ giảm, và ngược lại.
  - Tốc độ học (Learning Rate - α): Đây là một siêu tham số quan trọng kiểm soát kích
thước bước trong mỗi lần cập nhật tham số:
    - Nếu tốc độ học quá nhỏ, mô hình sẽ hội tụ rất chậm.
    - Nếu tốc độ học quá lớn, mô hình có thể "vượt qua" điểm cực tiểu tối ưu hoặc
thậm chí phân kỳ (không ổn định).

### 3. Backpropagation: Tính toán Gradient một cách Hiệu quả.
  - Backpropagation (lan truyền ngược) là thuật toán được sử dụng để tính toán gradient
của hàm mất mát đối với tất cả các trọng số và độ lệch của mạng một cách hiệu quả.5
Nó hoạt động bằng cách lan truyền lỗi ngược từ lớp đầu ra đến lớp đầu vào, sử dụng
quy tắc chuỗi của giải tích để tính toán gradient cho từng lớp.5
  - Backpropagation làm cho Gradient Descent khả thi đối với các mạng đa lớp bằng
cách cung cấp một cách hiệu quả để có được các gradient cần thiết cho việc cập
nhật tham số.11 Nếu không có Backpropagation, việc tính toán gradient cho các mạng
sâu sẽ cực kỳ tốn kém về mặt tính toán:
Các vấn đề thường gặp trong quá trình backpropagation và huấn luyện bao gồm:
    - Mất mát Gradient (Vanishing Gradients): Gradient của các lớp thấp hơn (gần
lớp đầu vào hơn) có thể trở nên rất nhỏ. Trong các mạng sâu, việc tính toán các
gradient này có thể liên quan đến việc nhân nhiều số hạng nhỏ với nhau. Khi giá
trị gradient tiến gần đến 0 đối với các lớp thấp hơn, các lớp đó sẽ học rất chậm
hoặc hoàn toàn không học được. Hàm kích hoạt ReLU có thể giúp ngăn chặn hiện
tượng mất mát gradient.
    - Bùng nổ Gradient (Exploding Gradients): Nếu các trọng số trong mạng rất lớn,
thì gradient của các lớp thấp hơn sẽ liên quan đến tích của nhiều số hạng lớn.
Trong trường hợp này, gradient có thể trở nên quá lớn để hội tụ. Chuẩn hóa theo
lô (Batch Normalization) hoặc giảm tốc độ học có thể giúp ngăn chặn hiện tượng
bùng nổ gradient.


# II. Các Khái niệm Cơ bản về PyTorch cho Học sâu
### A. torch.Tensor: Khối Xây dựng của PyTorch
- Là cấu trúc dữ liệu cơ bản nhất trong PyTorch, tương tự như ndarray trong NumPy, nhưng có thể sử dụng trên GPU để tăng tốc tính toán.
- Các thuộc tính chính của Tensor: Các thuộc tính của Tensor mô tả hình dạng, kiểu dữ liệu và thiết bị lưu trữ của chúng.
    ```
    tensor = torch.rand(3,4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    ```
  - tensor.shape: Trả về một torch.Size (một tuple) mô tả số chiều của Tensor.16
  - tensor.dtype: Kiểu dữ liệu của các phần tử trong Tensor (ví dụ: torch.float32,
torch.int64).16
  - tensor.device: Thiết bị mà Tensor được lưu trữ trên đó (ví dụ: 'cpu', 'cuda:0'). Mặc định, tensor được tạo trên CPU. Để di chuyển tensor sang GPU (nếu có), có thể sử dụng phương thức .to(): tensor = tensor.to('cuda'). Việc sao chép tensor lớn giữa các thiết bị có thể tốn kém về thời gian và bộ nhớ.
- Các phép toán chính trên Tensor:
  - PyTorch cung cấp hơn 100 phép toán trên tensor, bao gồm số học, đại số tuyến tính, thao tác ma trận (chuyển vị, lập chỉ mục, cắt lát), lấy mẫu, v.v..
  - Lập chỉ mục và Cắt lát (Indexing and Slicing): Tương tự như NumPy.
```
    tensor = torch.ones(4, 4)
    print(f"Hàng đầu tiên: {tensor}")print(f"Cột đầu tiên: {tensor[:, 0]}")
    print(f"Cột cuối cùng: {tensor[..., -1]}")
    tensor[:,1] = 0 # Gán giá trị cho một cột
```
  - Phép toán số học: Cộng, trừ, nhân, chia theo phần tử; nhân ma trận (@ hoặc torch.matmul).
```
    Nhân ma trận
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    Nhân theo phần tử
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
```
  - Phép toán tại chỗ (In-place operations): Các phép toán lưu kết quả vào toán hạng được gọi là phép toán tại chỗ. Chúng được biểu thị bằng hậu tố _. Ví dụ: x.copy_(y), x.t_() sẽ thay đổi x.
```
    print(tensor)
    tensor.add_(5) # Cộng 5 vào mỗi phần tử của tensor và lưu tại chỗ
    print(tensor)
```
  - Tổng hợp (Aggregation): Ví dụ: tensor.sum(). Để chuyển đổi Tensor một phần tử thành một số Python, sử dụng item().
```
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
```
  - Nối (Concatenation): torch.cat([tensor1, tensor2], dim=...) để nối các tensor
theo một chiều nhất định.17
- Cầu nối với NumPy:
  - Tensor sang mảng NumPy: tensor.numpy(). Nếu Tensor nằm trên CPU, mảng NumPy và Tensor sẽ chia sẻ cùng một vùng nhớ.
  - Mảng NumPy sang Tensor:  torch.from_numpy(np_array). Tương tự, chúng chia sẻ bộ nhớ.

### B. Quản lý Dữ liệu: Dataset và DataLoader

- torch.utils.data.Dataset:
  - Dataset là một lớp trừu tượng trong PyTorch dùng để biểu diễn một tập dữ liệu. Khi làm việc với dữ liệu tùy chỉnh, người dùng cần tạo một lớp con kế thừa từ Dataset và ghi đè hai phương thức chính :
    - __len__(self): Phương thức này phải trả về kích thước (số lượng mẫu) của tập dữ liệu. Ví dụ, len(dataset) sẽ gọi phương thức này.
    - __getitem__(self, idx): Phương thức này hỗ trợ việc lập chỉ mục, cho phép truy xuất mẫu thứ idx từ tập dữ liệu (ví dụ: dataset[i]). Nó thường chịu trách nhiệm tải và trả về một mẫu dữ liệu (ví dụ: một cặp hình ảnh và nhãn của nó).
    from torch.utils.data import Dataset

    ```
    class MyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    ```
- torch.utils.data.DataLoader:
  - DataLoader là một trình vòng lặp (iterator) bao bọc một đối tượng Dataset để cung
cấp một cách thuận tiện để lặp qua các mẫu dữ liệu.19 Nó cung cấp các chức năng
quan trọng như:
  - Tải dữ liệu theo lô (Batching the data): batch_size là một tham số quan trọng,xác định số lượng mẫu được sử dụng trong một lần lặp (một lô). Việc xử lý dữ liệu theo lô giúp tận dụng khả năng tính toán song song của phần cứng và có thể giúp ổn định quá trình huấn luyện.
  - Xáo trộn dữ liệu (Shuffling the data): Tham số shuffle (Boolean True/False) xác định liệu dữ liệu có nên được xáo trộn sau mỗi epoch (một lượt duyệt qua toàn bộ tập dữ liệu) hay không.21 Xáo trộn dữ liệu giúp ngăn mô hình học các phụ thuộc thứ tự không mong muốn trong dữ liệu và cải thiện khả năng tổng quát hóa.
  - Tải dữ liệu song song (Parallel data loading): Tham số num_workers (số nguyên) chỉ định số lượng tiến trình con được sử dụng để tải dữ liệu.21 Giá trị num_workers > 0 cho phép tải dữ liệu song song với quá trình huấn luyện, giúp giảm thiểu tắc nghẽn do tải dữ liệu và tăng tốc độ huấn luyện tổng thể.
  - Ghim bộ nhớ (Pinned memory): Tham số pin_memory (Boolean True/False). Nếu là True, DataLoader sẽ sao chép Tensor vào bộ nhớ CUDA được ghim (pinned memory) trước khi trả về chúng.22 Điều này có thể tăng tốc độ truyền dữ liệu từ CPU sang GPU.
  - Hàm tập hợp lô (Collate function): Tham số collate_fn cho phép người dùng chỉ định một hàm tùy chỉnh để hợp nhất một danh sách các mẫu thành một lô nhỏ.Điều này hữu ích khi các mẫu có cấu trúc phức tạp hoặc cần xử lý đặc biệt khi tạo lô.
    ``` 
       from torch.utils.data import DataLoader
        dataset = MyDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    ```

### C. Định nghĩa Mô hình: torch.nn.Module
- nn.Module là lớp cơ sở cho tất cả các module mạng nơ-ron, từ các lớp đơn giản như nn.Linear đến toàn bộ các mô hình phức tạp. Để tạo một mô hình tùy chỉnh, người dùng định nghĩa một lớp mới kế thừa từ nn.Module.
- Cấu trúc của một lớp nn.Module tùy chỉnh:
  - Phương thức __init__(self):
    - Đây là nơi khởi tạo các thành phần của mạng. Điều quan trọng là phải gọi super().__init__() ở đầu phương thức này để khởi tạo lớp nn.Module cơ sở.23
    - Trong __init__, người dùng định nghĩa các lớp (layers) của mạng (ví dụ: self.fc1 = nn.Linear(in_features, out_features), self.conv1 = nn.Conv2d(...)) và các tham số khác cần thiết. Các lớp này bản thân chúng cũng là các thực thể của nn.Module.20
    - Các tham số được định nghĩa dưới dạng thuộc tính của nn.Module (hoặc được bọc trong nn.Parameter) sẽ tự động được đăng ký với mô hình. Điều này có nghĩa là PyTorch sẽ theo dõi chúng, và chúng có thể được truy cập thông qua các phương thức như model.parameters() để sử dụng trong trình tối ưu hóa.20
```
    import torch
    import torch.nn as nn
    import math
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    Các tham số trong nn.Linear (weights, biases) được tự động đăng ký
    self.my_custom_param = nn.Parameter(torch.randn(10, 5)) # Ví dụ tham số tùy chỉnh
```
  - Phương thức forward(self, input_data):
    - Phương thức này định nghĩa quá trình truyền thẳng (forward pass) của mô hình: cách dữ liệu đầu vào (input_data) chảy qua các lớp đã được định nghĩa trong __init__ để tạo ra một đầu ra.
    - Khi một thực thể của mô hình được gọi (ví dụ: output = model(input_tensor)), PyTorch sẽ tự động thực thi phương thức forward của nó.
```
    def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
```

### D. Tối ưu hóa: torch.optim.Optimizer
- torch.optim triển khai các thuật toán tối ưu hóa khác nhau được sử dụng để cập nhật các tham số (trọng số và độ lệch) của mô hình trong quá trình huấn luyện
```
    import torch.optim as optim
    model = MyModel()
    # SGD (Stochastic Gradient Descent)
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
```
- Sử dụng Optimizer trong Vòng lặp Huấn luyện:
  - Quá trình tối ưu hóa trong vòng lặp huấn luyện thường bao gồm ba bước chính liên quan đến optimizer:
    - optimizer.zero_grad(): Đặt lại gradient của tất cả các tham số đã được tối ưu hóa về 0. Điều này rất quan trọng vì theo mặc định, PyTorch cộng dồn gradient sau mỗi lần gọi loss.backward(). Do đó, zero_grad() thường được gọi ở đầu mỗi lần lặp huấn luyện, trước khi tính toán gradient mới.
    - loss.backward(): Tính toán gradient của hàm mất mát đối với các tham số của mô hình. (Điều này được thực hiện bởi Autograd, sẽ được thảo luận sau).
    - optimizer.step(): Cập nhật giá trị của các tham số dựa trên gradient đã được tính toán và lưu trữ trong thuộc tính .grad của mỗi tham số. Phương thức step() thực hiện logic cập nhật cụ thể của thuật toán tối ưu hóa đã chọn (ví dụ: quy tắc cập nhật SGD hoặc Adam).

### E. Định lượng Lỗi: torch.nn.LossFunction
-  torch.nn được sử dụng để đo lường mức độ khác biệt (sai số) giữa đầu ra dự đoán của mô hình và giá
trị mục tiêu thực tế 
```
    import torch.nn as nn
    # Khởi tạo hàm mất mát
    # Đối với bài toán hồi quy
    mse_loss_fn = nn.MSELoss()
    # Đối với bài toán phân loại đa lớp
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    # Giả sử 'predictions' là đầu ra của mô hình và 'targets' là nhãn thực tế
    # Tính toán loss
    # loss_value_mse = mse_loss_fn(predictions_regression, targets_regression)
    # loss_value_ce = cross_entropy_loss_fn(predictions_classification_logits, targets_classification_labels)
```
- Một số hàm mất mát phổ biến trong torch.nn bao gồm:
  - nn.MSELoss: Sai số toàn phương trung bình, thường dùng cho các tác vụ hồi quy.
  - nn.CrossEntropyLoss: Kết hợp nn.LogSoftmax và nn.NLLLoss trong một lớp duy nhất. Nó thường được sử dụng cho các tác vụ phân loại đa lớp và mong đợi đầu ra thô (logits) từ mô hình, vì nó sẽ tự áp dụng LogSoftmax bên trong.
  - nn.NLLLoss (Negative Log Likelihood Loss): Thường được sử dụng kết hợp với một lớp nn.LogSoftmax ở cuối mô hình cho các tác vụ phân loại.13
  - nn.BCELoss (Binary Cross Entropy Loss): Dùng cho các tác vụ phân loại nhị phân

### F. Tự động Tính toán Đạo hàm: torch.autograd
- torch.autograd là công cụ cho phép tự động tính toán đạo hàm, đây là
nền tảng cho thuật toán backpropagation trong huấn luyện mạng nơ-ron.
- Thuộc tính requires_grad:
  - Mỗi Tensor trong PyTorch có một thuộc tính boolean là requires_grad. Nếu requires_grad được đặt thành True cho một Tensor, autograd sẽ theo dõi tất cả các phép toán được thực hiện trên Tensor đó.18 Các tham số của mô hình (được tạo từ nn.Parameter hoặc các lớp như nn.Linear, nn.Conv2d) mặc định có requires_grad=True.
  - Đồ thị Tính toán (Computational Graph - DAG): autograd ghi lại các phép toán trong một Đồ thị Không Tuần hoàn Có Hướng (Directed Acyclic Graph - DAG). Trong DAG này:
    - Các nút lá (leaves) là các Tensor đầu vào (ví dụ: dữ liệu đầu vào, các tham số của mô hình).
    - Các nút gốc (roots) là các Tensor đầu ra (ví dụ: giá trị mất mát). Trong quá trìnhtruyền thẳng, autograd thực hiện hai việc đồng thời:
  - Chạy phép toán được yêu cầu để tính toán Tensor kết quả.
  - Duy trì hàm tính gradient của phép toán đó trong DAG. Một điểm quan trọng là đồ thị này được tạo động từ đầu sau mỗi lần gọi .backward().Điều này cho phép sử dụng các câu lệnh điều khiển luồng (ví dụ: vòng lặp, câu lệnh if) trong mô hình, cho phép thay đổi hình dạng, kích thước và các phép toán ở mỗi lần lặp nếu cần.
- loss.backward():
  - Khi .backward() được gọi trên một Tensor vô hướng (thường là giá trị mất mát loss), autograd sẽ bắt đầu quá trình backpropagation.
    - autograd tính toán gradient của loss đối với tất cả các Tensor trong đồ thị có requires_grad=True (ví dụ: các tham số của mô hình) bằng cách sử dụng quy tắc chuỗi.
    - Các gradient này được tích lũy (cộng dồn) vào thuộc tính .grad của các Tensor tương ứng (ví dụ: weights.grad, bias.grad).
- torch.no_grad():
    - Đây là một trình quản lý ngữ cảnh (context manager) được sử dụng để tạm thời vô hiệu hóa việc theo dõi gradient cho các phép toán bên trong khối lệnh của nó. Điều này hữu ích trong giai đoạn suy luận (inference), khi không cần tính gradient, hoặc khi cập nhật trọng số thủ công (như trong một số thuật toán tối ưu hóa tùy chỉnh) để ngăn autograd theo dõi các phép cập nhật này

# III. Mạng Nơ-ron Tích chập (Convolutional Neural Networks -CNNs)
## A. Kiến trúc Cốt lõi của CNNs
- Mạng Nơ-ron Tích chập (CNNs hay ConvNets) là một loại mô hình học sâu đặc biệt hiệu quả đối với dữ liệu có cấu trúc dạng lưới. Chúng được sử dụng rộng rãi cho các tác vụ như nhận dạng hình ảnh, xử lý hình ảnh và phân loại hình ảnh.
### 1. Lớp Tích chập (Convolutional Layer): Bộ lọc, Bản đồ Đặc trưng, Stride, Padding, Trọng số Chia sẻ.    
- Lớp tích chập là nền tảng của CNN, chịu trách nhiệm chính trong việc trích xuất đặc trưng từ dữ liệu đầu vào.
- Tensor Đầu vào: Hình ảnh thường là tensor 3D (kênh màu, chiều cao, chiều rộng) hoặc 4D (thêm chiều lô).
- Bộ lọc/Nhân (Filters/Kernels): Ma trận nhỏ chứa trọng số học được, trượt ("tích chập") trên đầu vào để phát hiện đặc trưng (cạnh, góc, kết cấu, mẫu phức tạp). Nhiều bộ lọc được dùng để tìm các đặc trưng khác nhau.
- Bản đồ Đặc trưng/Kích hoạt (Feature/Activation Map): Đầu ra của một bộ lọc, biểu thị sự hiện diện và cường độ của đặc trưng tại các vị trí không gian.
- Stride (Bước nhảy): Kích thước bước bộ lọc di chuyển. Stride > 1 làm giảm kích thước bản đồ đặc trưng. Công thức kích thước đầu ra: ⌊(N−F+2P)/S⌋+1 (N: kích thước đầu vào, F: kích thước bộ lọc, P: padding, S: stride).
- Padding (Đệm): Thêm pixel (thường là 0) quanh biên ảnh đầu vào.
  - Valid Padding: Không padding, kích thước đầu ra thu nhỏ.
  - Same Padding: Padding để kích thước đầu ra bằng đầu vào (với stride 1).
- Lợi ích: Giữ kích thước không gian, xử lý biên hiệu quả, tránh mất thông tin rìa.
- Trọng số Chia sẻ (Shared Weights): Đặc điểm mạnh mẽ của CNN. Trọng số của một bộ lọc được dùng trên toàn bộ đầu vào.
  - Lợi ích: Giảm đáng kể số lượng tham số, tạo khả năng bất biến với dịch chuyển (translation invariance), giúp tránh vấn đề gradient. Việc chia sẻ trọng số dựa trên giả định rằng đặc trưng hữu ích ở một phần ảnh cũng hữu ích ở phần khác. CNN trích xuất đặc trưng theo cấp bậc, tương tự vỏ não thị giác người.

### 2. Lớp Gộp (Pooling Layer): Max Pooling, Average Pooling để Giảm chiều Dữ liệu.
- Mục đích: Giảm chiều không gian (giảm tham số, tính toán), kiểm soát quá khớp (overfitting), tạo tính bất biến với dịch chuyển cục bộ.
- Hoạt động: Áp dụng hàm gộp trên cửa sổ nhỏ (ví dụ: 2×2) trượt qua bản đồ đặc trưng.
- Loại phổ biến:
  - Max Pooling: Chọn giá trị lớn nhất trong vùng cục bộ (giữ đặc trưng nổi bật).
  - Average Pooling: Tính giá trị trung bình (biểu diễn tổng quát, mượt hơn). Mặc dù phổ biến, một số kiến trúc hiện đại giảm sử dụng pooling (ví dụ: dùng stride lớn ở lớp tích chập).
### 3. Lớp Duỗi thẳng (Flatten Layer): Chuẩn bị cho Phân loại.
- Mục đích của lớp này là chuyển đổi định dạng dữ liệu từ dạng lưới (phù hợp cho các lớptích chập và gộp) sang dạng vector phẳng, là đầu vào tiêu chuẩn cho các lớp kết nối đầy đủ (fully connected layers).
- Lớp duỗi thẳng hoạt động như một cầu nối giữa phần trích xuất đặc trưng (các lớp tích chập/gộp) và phần phân loại/ra quyết định (các lớp kết nối đầy đủ) của một CNN. 

### 4. Lớp Kết nối Đầy đủ (Fully Connected Layer): Suy luận Cấp cao và Đầu ra.
- Còn được gọi là các lớp dày đặc (dense layers),thường được tìm thấy ở cuối kiến trúc CNN, sau khi các đặc trưng đã được trích xuất bởi các lớp tích chập và gộp, và sau đó được duỗi thẳng.
- Cấu trúc: Mỗi nơ-ron kết nối với tất cả kích hoạt lớp trước (giống Perceptron Đa Lớp - MLP).
- Chức năng: Phân loại hoặc hồi quy dựa trên đặc trưng cấp cao đã học.
- Lớp Đầu ra: Dùng hàm kích hoạt phù hợp (ví dụ: Softmax cho phân loại đa lớp). Các lớp FC kết hợp đặc trưng cục bộ để đưa ra quyết định toàn cục. Chúng có thể chứa nhiều tham số, dễ bị quá khớp (thường dùng Dropout để giảm thiể

## B. Ứng dụng của CNNs trong Xử lý Hình ảnh
- CNNs đã cách mạng hóa thị giác máy tính.
### 1. Phân loại Hình ảnh (Image Classification): 
- Gán một nhãn cho toàn bộ hình ảnh (ví dụ: "mèo", "chó"). CNN tự động học hệ thống phân cấp các đặc trưng.
  - Tập dữ liệu tiêu chuẩn: MNIST (chữ số viết tay 28×28), CIFAR-10 (10 lớp ảnh màu 32×32), ImageNet (triệu ảnh, ngàn lớp).
  - Kiến trúc điển hình: Chuỗi [Lớp Tích chập - ReLU - Lớp Gộp], sau đó Lớp Duỗi thẳng, và một/nhiều Lớp Kết nối Đầy đủ (Softmax ở lớp cuối).
  - Thành công: AlexNet (2012) trên ImageNet là bước ngoặt. Học chuyển giao (transfer learning) rất hiệu quả, sử dụng CNN tiền huấn luyện trên tập lớn (như ImageNet) rồi tinh chỉnh cho tập dữ liệu mới nhỏ hơn.

### 2. Phát hiện Đối tượng (Object Detection):
- Xác định các đối tượng và định vị chúng (bằng hộp giới hạn - bounding boxes). Bao gồm phân loại và định vị.
- Quá trình phát triển:
  - R-CNN (Region-based CNN): Dùng thuật toán ngoài (Selective Search) đề xuất vùng (RoIs), CNN trích xuất đặc trưng cho từng vùng, SVM phân loại. Chậm.
  - Fast R-CNN: Cho toàn bộ ảnh qua CNN một lần, chiếu vùng đề xuất lên bản đồ đặc trưng chung, lớp RoI Pooling trích xuất đặc trưng cố định kích thước. Nhanh hơn.
  - Faster R-CNN: Giới thiệu Mạng Đề xuất Vùng (Region Proposal Network - RPN) tự đề xuất RoIs, làm quá trình end-to-end và nhanh hơn nhiều.
  - YOLO (You Only Look Once): Tiếp cận như bài toán hồi quy đơn lẻ, chia ảnh thành lưới, mỗi ô dự đoán hộp giới hạn và xác suất lớp. Rất nhanh, phù hợp thời gian thực.
  - SSD (Single Shot MultiBox Detector): Tương tự YOLO, phát hiện đối tượng ở nhiều tỷ lệ và kích thước từ nhiều lớp tích chập. Cân bằng tốc độ và độ chính xác.
- Tập dữ liệu: COCO (Common Objects in Context). Xu hướng phát triển hướng tới học end-to-end và tối ưu tốc độ. Phát hiện đối tượng phức tạp hơn phân loại, đòi hỏi kiến trúc và hàm mất mát phức tạp hơn (kết hợp mất mát phân loại và định vị). Thành công mở đường cho xe tự lái, robot.

# IV. Mạng Nơ-ron Hồi quy (RNNs) & Bộ nhớ Dài-Ngắn hạn (Long Short-Term Memory - LSTM) cho Dữ liệu Tuần tự

## A. Mạng Nơ-ron Hồi quy (RNNs)
### 1. Khái niệm về Tính Hồi quy và Bộ nhớ trong Chuỗi
- Mạng Nơ-ron Hồi quy (RNNs) là một lớp mạng nơ-ron được thiết kế đặc biệt để xử lý
dữ liệu tuần tự (sequential data), nơi thứ tự của các phần tử là quan trọng. Ví dụ về dữ
liệu tuần tự bao gồm chuỗi thời gian (time series), văn bản ngôn ngữ tự nhiên, tín hiệu
giọng nói, v.v..
- Ý tưởng cốt lõi của RNN là chúng có một "vòng lặp hồi quy" hoặc "kết nối phản hồi (feedback loop). Điều này có nghĩa là đầu ra từ một bước thời gian (time step) trước đó được đưa trở lại làm đầu vào cho bước thời gian hiện tại, cho phép mạng duy trì một "bộ nhớ" nội bộ hoặc "trạng thái ẩn" (hidden state).1 Trạng thái ẩn này tóm tắt thông tin từ tất cả các bước thời gian trước đó trong chuỗi.
- Có nhiều loại kiến trúc RNN dựa trên chuỗi đầu vào/đầu ra:
  - Một-đến-nhiều (One-to-many): Một đầu vào tạo ra một chuỗi đầu ra (ví dụ: tạo chú thích cho hình ảnh).
  - Nhiều-đến-một (Many-to-one): Một chuỗi đầu vào tạo ra một đầu ra duy nhất (ví dụ: phân tích cảm xúc của một câu).
  - Nhiều-đến-nhiều (Many-to-many): Một chuỗi đầu vào tạo ra một chuỗi đầu ra (ví dụ: dịch máy, nơi độ dài chuỗi đầu vào và đầu ra có thể khác nhau).

### 2. Vấn đề Mất mát Gradient trong RNNs.
- RNNs gặp khó khăn trong việc học các "phụ thuộc dài hạn" (mối quan hệ giữa các phần tử xa nhau) do vấn đề mất mát gradient (vanishing gradient) hoặc bùng nổ gradient (exploding gradient).
- Nguyên nhân: Khi huấn luyện bằng Backpropagation Through Time (BPTT), RNN được "mở rộng" thành mạng sâu. Gradient lan truyền ngược qua nhiều bước, nhân lặp đi lặp lại các ma trận trọng số W và đạo hàm hàm kích hoạt. Nếu các giá trị này nhỏ, gradient sẽ co lại theo cấp số nhân ("biến mất"); nếu lớn, gradient sẽ tăng vọt ("bùng nổ").
- Tác động: Mất mát gradient làm cho các trọng số liên quan đến các bước thời gian xa không được cập nhật, khiến mạng chỉ có "bộ nhớ ngắn hạn". Bùng nổ gradient gây mất ổn định (có thể giải quyết bằng "cắt gradient" - gradient clipping). Vấn đề này là rào cản lớn cho RNN, thúc đẩy sự phát triển của LSTM và GRU.

## B. Mạng Bộ nhớ Dài-Ngắn hạn (Long Short-Term Memory - LSTM)
### 1. Kiến trúc: Trạng thái Ô nhớ, Cổng Quên, Cổng Đầu vào, Cổng Đầu ra.
- Mạng Bộ nhớ Dài-Ngắn hạn (LSTM) là một loại RNN đặc biệt, được thiết kế rõ ràng để tránh vấn đề phụ thuộc dài hạn bằng cách có khả năng ghi nhớ thông tin trong thời gian dài.
- Trạng thái Ô nhớ (Ct): Điểm mấu chốt, hoạt động như một "băng chuyền" mang thông tin qua chuỗi với ít thay đổi, cho phép lưu trữ thông tin dài hạn.Thông tin có thể dễ dàng chảy dọc theo nó mà không bị thay đổi nhiều LSTM có khả năng thêm hoặc bớt thông tin vào trạng thái ô nhớ, được điều chỉnh cẩn thận bởi các cấu trúc gọi là cổng (gates)
- Một LSTM có ba loại cổng chính để bảo vệ và kiểm soát trạng thái ô nhớ:
  - Cổng Quên (Forget Gate - ft): Quyết định thông tin nào sẽ bị loại bỏ khỏi trạng thái ô nhớ trước đó Ct−1
  - Cổng Đầu vào (Input Gate - it và gt hoặc C~t): Quyết định thông tin mới nào sẽ được lưu trữ trong trạng thái ô nhớ.
  - Cập nhật Trạng thái Ô nhớ: Trạng thái ô nhớ mới Ct được tính bằng cách nhân trạng thái cũ Ct−1 với cổng quên ft (để loại bỏ thông tin không cần thiết), sau đó cộng với tích của cổng đầu vào it và các giá trị ứng cử viên mới gt (để thêm thông tin mới).
  - Cổng Đầu ra (Output Gate - ot): Quyết định đầu ra (trạng thái ẩn ht) sẽ là gì, dựa trên trạng thái ô nhớ Ct đã được cập nhật.

### 2. Cách LSTM Giảm thiểu Mất mát Gradient và Nắm bắt Phụ thuộc Dài hạn.
- Kiến trúc phức tạp của LSTM với trạng thái ô nhớ và các cổng giúp nó giải quyết hiệu
quả vấn đề mất mát gradient và nắm bắt các phụ thuộc dài hạn trong dữ liệu tuần tự:
  - Cập nhật Trạng thái Ô nhớ Cộng dồn (Additive Cell State Update): Khía cạnh quan trọng nhất là việc cập nhật trạng thái ô nhớ mang tính cộng dồn: . Đường dẫn cộng dồn này cho phép gradient chảy ngược mà ít bị cản trở (đặc biệt nếu ft≈1), gọi là "đường cao tốc gradient", tránh nhân lặp các ma trận trọng số nhỏ.
  - Kiểm soát bằng Cổng: Các cổng điều khiển luồng thông tin và gradient, đảm bảo chỉ thông tin liên quan ảnh hưởng Ct và ht , bảo vệ các phụ thuộc dài hạn. Thuật ngữ "Vòng quay lỗi không đổi" (Constant Error Carousel - CEC) mô tả khả năng này. Mặc dù LSTM giảm thiểu đáng kể mất mát gradient, chúng phức tạp hơn RNN đơn giản và vẫn có thể gặp vấn đề với chuỗi cực dài.

## C. Ứng dụng của RNNs và LSTMs trong Xử lý Chuỗi  
### 1.Xử lý Văn bản (NLP - Natural Language Processing):
- Mô hình hóa Ngôn ngữ: Dự đoán từ/ký tự tiếp theo.
- Dịch máy: Kiến trúc bộ mã hóa - bộ giải mã (encoder-decoder). Encoder (LSTM) nén câu đầu vào thành vector ngữ cảnh; Decoder (LSTM) tạo câu dịch từ vector đó.
- Cơ chế Chú ý (Attention Mechanism): Cho phép decoder "chú ý" đến các phần khác nhau của câu đầu vào tại mỗi bước dịch, cải thiện dịch câu dài, cách mạng hóa dịch máy thần kinh.
- Phân tích Cảm xúc: Xác định cảm xúc (tích cực/tiêu cực) từ văn bản (thường là kiến trúc nhiều-đến-một).
- Tạo Văn bản: Tạo văn bản mới mạch lạc (ví dụ: cấp độ ký tự bằng LSTM). Mô hình học các mẫu và ngữ cảnh từ dữ liệu huấn luyện.

### 2.Phân tích Dữ liệu Chuỗi Thời gian (Time-Series Data Analysis):

- Dự báo: Dự đoán giá trị tương lai (giá cổ phiếu, thời tiết, nhu cầu năng lượng).
- Phương pháp dự báo với LSTM:
- Vòng lặp Mở (Open-loop): Dùng dữ liệu thực tế làm đầu vào cho mỗi dự đoán.
- Vòng lặp Kín (Closed-loop): Dùng dự đoán trước đó làm đầu vào cho dự đoán tiếp theo (dự báo nhiều bước).
- Ví dụ: Dự báo Giá Cổ phiếu: LSTM huấn luyện trên dữ liệu giá lịch sử để dự đoán giá tương lai, có khả năng "ghi nhớ" các sự kiện/xu hướng trong quá khứ. Thường vượt trội hơn các mô hình thống kê truyền thống (ARIMA) do khả năng nắm bắt các phụ thuộc phi tuyến và tính không dừng.