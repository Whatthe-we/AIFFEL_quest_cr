import 'package:flutter/material.dart';
import 'SecondPage.dart';

class FirstPage extends StatefulWidget {
  @override
  _FirstPageState createState() => _FirstPageState();
}

class _FirstPageState extends State<FirstPage> {
  bool isCat = true;
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
              'assets/free-icon-cat.png',
              fit: BoxFit.contain,
            ),
          ),
        ),
        title: Container(
          color: Colors.black,
          child: Text(
            "First Page",
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        centerTitle: true, // 타이틀을 중앙에 배치
        actions: [
          IconButton(
            icon: Icon(
              Icons.favorite,
              color: Colors.white, // 하트 아이콘을 흰색으로 설정
            ),
            onPressed: () {
              // 하트 아이콘 클릭 시 동작할 코드 작성
              print("야옹");
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
                setState(() {
                  isCat = false;
                });
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => SecondPage(isCat: isCat),
                  ),
                );
              },
              child: Text("Next"),
            ),
          ),
          GestureDetector(
            onTap: () {
              setState(() {
                isCat = true; // 고양이 이미지가 눌렸을 때 isCat을 true로 설정
              });
              print("isCat 상태: $isCat");

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
                'assets/cat.png', // Local image for cat
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
