import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/ai_screen.dart';
import 'screens/categories_screen.dart';
import 'screens/profile_screen.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SplashScreen(),  // SplashScreen으로 시작
      localizationsDelegates: [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: [
        Locale('ko', 'KR'), // 한국어 지원
      ],
      theme: ThemeData(
        // 앱 전체에 사용할 폰트 설정
        fontFamily: 'Jua',  // Jua 폰트 적용
      ),
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // 2초 후에 홈 화면으로 이동
    Future.delayed(Duration(seconds: 2), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => BottomNavBar()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Image.asset(
          'assets/aaa.png',
          fit: BoxFit.contain,  // 이미지를 화면에 맞게 조정
          width: 500,  // 원하는 너비로 설정
          height: 500,  // 원하는 높이로 설정
        ),
      ),
    );
  }
}

class BottomNavBar extends StatefulWidget {
  @override
  _BottomNavBarState createState() => _BottomNavBarState();
}

class _BottomNavBarState extends State<BottomNavBar> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    HomeScreen(),
    AIScreen(),
    CategoriesScreen(),
    ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        // Text 대신 Image 사용
        title: Align(
          alignment: Alignment.centerLeft,  // 왼쪽 정렬
          child: Image.asset(
            'assets/aaa.png',  // 사용할 이미지 경로
            height: 150,  // 이미지 높이 조정
            width: 150,   // 이미지 너비 조정
          ),
        ),
        backgroundColor: Color(0xFFFFE4B5),
      ),
      body: _screens[_currentIndex],  // 화면 전환
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,  // 현재 탭 인덱스
        onTap: (index) {
          setState(() {
            _currentIndex = index;  // 탭 클릭 시 화면 전환
          });
        },
        backgroundColor: Colors.white,
        selectedItemColor: Color(0xFFFFA500),
        unselectedItemColor: Colors.grey[600],  // 선택되지 않은 항목 색상 변경
        selectedFontSize: 14,  // 선택된 아이템 글자 크기
        unselectedFontSize: 12,  // 선택되지 않은 아이템 글자 크기
        showUnselectedLabels: true,  // 선택되지 않은 아이템 라벨 표시
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home, size: 28),  // 아이콘 크기 조정
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.device_hub, size: 28),  // AI 도우미 아이콘
            label: 'AI 도우미',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.category, size: 28),  // 카테고리 아이콘
            label: '카테고리',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.account_circle, size: 28),  // 마이페이지 아이콘
            label: '마이페이지',
          ),
        ],
      ),
    );
  }
}