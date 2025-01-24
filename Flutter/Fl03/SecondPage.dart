import 'package:flutter/material.dart';

class SecondPage extends StatefulWidget {
  final bool isCat;

  const SecondPage({Key? key, required this.isCat}) : super(key: key);

  @override
  _SecondPageState createState() => _SecondPageState();
}

class _SecondPageState extends State<SecondPage> {
  bool showImage = false; // 이미지를 보일지 말지를 결정하는 변수

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.black,
        leading: Padding(
          padding: const EdgeInsets.all(8.0),
          child: ColorFiltered(
            colorFilter: ColorFilter.mode(
              Colors.white, // 아이콘을 흰색으로 변경
              BlendMode.srcIn,
            ),
            child: Image.asset(
              'assets/free-icon-dog.png', // 강아지 아이콘으로 변경
              fit: BoxFit.contain,
            ),
          ),
        ),
        title: Center(
          child: Text(
            "Second Page",
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(
              Icons.favorite,
              color: Colors.white, // 하트 아이콘을 흰색으로 설정
            ),
            onPressed: () {
              print("멍멍"); // 하트 아이콘 클릭 시 출력되는 메시지
            },
          ),
        ],
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Center(
            child: ElevatedButton(
              onPressed: () {
                Navigator.pop(context); // 이전 페이지로 돌아가기
              },
              child: Text("Back"),
            ),
          ),
          GestureDetector(
            onTap: () {
              print("isCat 상태: ${widget.isCat}");

              // 이미지를 보이게 설정
              setState(() {
                showImage = true;
              });

              // 0.1초 뒤에 이미지를 숨기도록 설정
              Future.delayed(Duration(milliseconds: 100), () {
                setState(() {
                  showImage = false;
                });
              });
            },
            child: Padding(
              padding: const EdgeInsets.only(top: 20.0),
              child: Image.asset(
                'assets/dog.jpg', // 강아지 이미지
                width: 300,
                height: 300,
              ),
            ),
          ),
          // aaa.png 이미지를 보였다가 사라지게 하기
          AnimatedOpacity(
            opacity: showImage ? 1.0 : 0.0, // showImage가 true일 때만 보이게 설정
            duration: Duration(milliseconds: 100),
            child: showImage
                ? Padding(
              padding: const EdgeInsets.only(top: 20.0),
              child: Image.asset(
                'assets/aaa.png', // Local image for aaa
                width: 100,
                height: 100,
              ),
            )
                : Container(), // showImage가 false일 때는 빈 컨테이너로 설정
          ),
        ],
      ),
    );
  }
}
