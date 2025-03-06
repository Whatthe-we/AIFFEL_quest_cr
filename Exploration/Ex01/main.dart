import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String result = "예측 결과가 여기에 표시됩니다.";
  Uint8List? selectedImageBytes;
  String? selectedImagePath; // 선택된 이미지 경로
  final TextEditingController urlController = TextEditingController(
    text: "https://1d63-35-233-243-255.ngrok-free.app/", // FastAPI 서버 URL
  );

  // 📌 사용 가능한 이미지 리스트
  final List<String> assetImages = [
    "assets/tulips.jpg",
    "assets/sunflowers.jpg",
    "assets/dandelion.PNG",
    "assets/daisy.jpg",
    "assets/roses.jpg",
  ];

  // 📌 라벨 ID와 꽃 이름 매핑
  final Map<String, String> labelMapping = {
    "0": "Dandelion",
    "1": "Daisy",
    "2": "Tulips",
    "3": "Sunflowers",
    "4": "Roses",
  };

  // 📌 assets에서 이미지를 로드하여 바이트 데이터로 변환
  Future<void> _loadImage(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    setState(() {
      selectedImageBytes = data.buffer.asUint8List();
      selectedImagePath = assetPath;
    });
  }

  // 📌 선택한 이미지 FastAPI 서버에 업로드하여 예측 요청
  Future<void> _uploadImage() async {
    if (selectedImageBytes == null) {
      setState(() {
        result = "이미지를 먼저 선택하세요!";
      });
      return;
    }

    try {
      final uri = Uri.parse(urlController.text + "predict");
      var request = http.MultipartRequest('POST', uri)
        ..headers['ngrok-skip-browser-warning'] = '69420'
        ..files.add(
          http.MultipartFile.fromBytes('image', selectedImageBytes!,
              filename: "selected_image.jpg"),
        );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        String predictedLabel = data['predicted_label'];
        String predictionScore = data['prediction_score'];

        // 라벨 ID를 꽃 이름으로 변환
        String flowerName = labelMapping[predictedLabel] ?? "Unknown";

        setState(() {
          result =
              "🔹 Predicted Flower: $flowerName\n📊 Prediction Score: $predictionScore";
        });
      } else {
        setState(() {
          result = "❌ 오류 발생: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        result = "⚠ 오류 발생: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Asset 이미지 예측")),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: <Widget>[
                TextField(
                  controller: urlController,
                  decoration: InputDecoration(labelText: "FastAPI 서버 URL"),
                ),
                SizedBox(height: 20),
                Text("📸 클릭하여 예측할 이미지를 선택하세요!", style: TextStyle(fontSize: 16)),
                SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: assetImages.map((path) {
                    return GestureDetector(
                      onTap: () => _loadImage(path),
                      child: Container(
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: selectedImagePath == path
                                ? Colors.blue
                                : Colors.transparent,
                            width: 3,
                          ),
                        ),
                        child: Image.asset(path, width: 100, height: 100),
                      ),
                    );
                  }).toList(),
                ),
                SizedBox(height: 20),
                selectedImageBytes != null
                    ? Column(
                        children: [
                          Text("선택된 이미지:", style: TextStyle(fontSize: 16)),
                          Image.memory(selectedImageBytes!,
                              width: 200, height: 200),
                          SizedBox(height: 20),
                          ElevatedButton(
                            onPressed: _uploadImage,
                            child: Text("📤 예측 요청"),
                          ),
                        ],
                      )
                    : Text("선택된 이미지 없음", style: TextStyle(fontSize: 16)),
                SizedBox(height: 20),
                Text(result, style: TextStyle(fontSize: 18)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
