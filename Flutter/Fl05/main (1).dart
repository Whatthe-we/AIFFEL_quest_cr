import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert'; // JSON 처리

void main() {
  runApp(JellyfishClassifierApp());
}

class JellyfishClassifierApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jellyfish Classifier',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
      ),
      home: ClassifierScreen(),
    );
  }
}

class ClassifierScreen extends StatelessWidget {
  // 예측 결과를 받아오는 함수
  Future<void> fetchPrediction() async {
    try {
      final response = await http.post(
        Uri.parse('http://<서버_IP>:<포트>/predict'), // 서버 주소와 포트 번호 수정
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'image': 'assets/jf.jpg'}), // 예시로 이미지 경로 전달
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        print('DEBUG: 예측 클래스는 ${data['class']}');
      } else {
        print('ERROR: 예측 실패');
      }
    } catch (e) {
      print('ERROR: 네트워크 요청 실패 - $e');
    }
  }

  // 예측 확률을 받아오는 함수
  Future<void> fetchPredictionProbability() async {
    try {
      final response = await http.post(
        Uri.parse('http://<서버_IP>:<포트>/predict_probability'), // 서버 주소와 포트 번호 수정
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'image': 'assets/jf.jpg'}), // 예시로 이미지 경로 전달
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        print('DEBUG: 예측 확률은 ${data['probability']}%');
      } else {
        print('ERROR: 예측 확률 요청 실패');
      }
    } catch (e) {
      print('ERROR: 네트워크 요청 실패 - $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.bubble_chart), // 해파리 아이콘 대체용
            SizedBox(width: 10),
            Text('Jellyfish Classifier'),
          ],
        ),
        centerTitle: false,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 업로드한 해파리 이미지 (300×300)
            Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.white, width: 2),
                borderRadius: BorderRadius.circular(8),
                image: DecorationImage(
                  image: AssetImage('assets/jf.jpg'), // 저장된 해파리 이미지
                  fit: BoxFit.cover,
                ),
              ),
            ),
            SizedBox(height: 20),
            // 좌측 및 우측 버튼
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // 좌측 버튼: '◁' 모양, 해파리 클래스 예측 결과 출력
                ElevatedButton(
                  onPressed: () {
                    fetchPrediction(); // 예측 클래스 요청
                  },
                  child: Text('◁', style: TextStyle(fontSize: 24)),
                ),
                SizedBox(width: 20),
                // 우측 버튼: '▷' 모양, 예측 확률 출력
                ElevatedButton(
                  onPressed: () {
                    fetchPredictionProbability(); // 예측 확률 요청
                  },
                  child: Text('▷', style: TextStyle(fontSize: 24)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// 회고: 이번에 서버가 연결이 안되서 결국 미완성된 걸 제출하고 말았다. 둘이 하던걸 혼자 하려니까 너무 힘들어서
// 아직 많이 부족하다는 걸 다시한번 느꼈다... 그리고 이번에 API 배울 때 유난히 오류, 문제가 많았는데
// 꼼꼼하게 하지 않아서 그런가... ㅠㅠ LMS상의 오류, 서버의 오류 등등 오류의 늪에 빠져서 제대로 퀘스트를
// 진행하지 못했다... 아쉽고 다음엔 더 열심히 해봐야겠다.
