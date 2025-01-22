import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());  // MyApp 위젯 실행해서 앱 시작
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),  // 홈 페이지 위젯 표시
    );
  }
}

class HomePage extends StatelessWidget {
  // 버튼 눌렸을 때 호출되는 메소드
  void handleButtonClick() {
    debugPrint('버튼이 눌렸습니다.');  // 버튼 클릭 시 콘솔에 메시지 출력
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blue,  // 앱바 배경색 파란색
        leading: Icon(Icons.favorite, color: Colors.white),  // 왼쪽에 하트 아이콘
        title: Text('플러터 앱 만들기'),  // 앱바 제목
        centerTitle: true,  // 제목 가운데 정렬
      ),
      body: Column(
        children: [
          // 첫 번째 Expanded: 버튼 화면 중앙에 배치
          Expanded(
            child: Center(
              child: ElevatedButton(
                onPressed: handleButtonClick,  // 버튼 클릭 시 handleButtonClick 호출
                child: Text('Text'),  // 버튼에 표시될 텍스트
              ),
            ),
          ),
          // 두 번째 Expanded: Stack 안에 여러 개의 Container 겹쳐서 배치
          Expanded(
            child: Center(
              child: Stack(
                alignment: Alignment.center,  // Stack 내부 위젯들을 중앙에 배치
                children: [
                  // 가장 큰 빨간색 Container
                  Container(
                    width: 300,
                    height: 300,
                    color: Colors.red,  // 빨간색
                  ),
                  // 오렌지색 Container (빨간색 위에 위치)
                  Positioned( // positioned 위젯으로 stack 안에서 특정 위치에 배치
                    top: 0,
                    left: 0,
                    child: Container(
                      width: 240,
                      height: 240,
                      color: Colors.orange,  // 오렌지색
                    ),
                  ),
                  // 노란색 Container (오렌지색 위에 위치)
                  Positioned(
                    top: 0,
                    left: 0,
                    child: Container(
                      width: 180,
                      height: 180,
                      color: Colors.yellow,  // 노란색
                    ),
                  ),
                  // 초록색 Container (노란색 위에 위치)
                  Positioned(
                    top: 0,
                    left: 0,
                    child: Container(
                      width: 120,
                      height: 120,
                      color: Colors.green,  // 초록색
                    ),
                  ),
                  // 파란색 Container
                  Positioned(
                    top: 0,
                    left: 0,
                    child: Container(
                      width: 60,
                      height: 60,
                      color: Colors.blue,  // 파란색
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// 최유진: 여러 요소를 겹쳐서 표현하는 경우에 어떻게 해야하는지에 대해서 알게 되었다.
// positioned 속성에 대해서 익히고 나니 더 창의적인 UI를 만들어 보고 싶다!
// 고명지: stack의 위치를 지정할 때 Positioned를 활용하여 각 기준점을 정해주는 방법을 배웠음.