import 'package:flutter/material.dart';
import 'category_detail_screen.dart'; // 상세 화면 파일 import

// 카테고리 화면
class CategoriesScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('카테고리')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2, // 2열로 그리드 배치
          crossAxisSpacing: 16.0, // 가로 간격
          mainAxisSpacing: 16.0, // 세로 간격
          children: <Widget>[
            categoryItem(Icons.home, '일상', context),
            categoryItem(Icons.home_repair_service, '주거', context),
            categoryItem(Icons.attach_money, '금융', context),
            categoryItem(Icons.health_and_safety, '건강', context),
            categoryItem(Icons.business, '커리어', context),
            categoryItem(Icons.insights, '보험', context),
            categoryItem(Icons.public, '공공', context),
            categoryItem(Icons.self_improvement, '자기개발', context), // 자기개발 추가
          ],
        ),
      ),
    );
  }

  // 카테고리 아이템을 클릭 시 새로운 화면으로 이동
  Widget categoryItem(IconData icon, String title, BuildContext context) {
    return GestureDetector(
      onTap: () {
        // 카테고리 클릭 시 해당 카테고리 정보를 보여주는 화면으로 이동
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => CategoryDetailScreen(title: title),
          ),
        );
      },
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(icon, size: 45), // 아이콘 크기 약간 줄임
            SizedBox(height: 8), // 텍스트와 아이콘 간격 조정
            Text(
              title,
              style: TextStyle(fontSize: 16), // 텍스트 크기 약간 줄임
            ),
          ],
        ),
      ),
    );
  }
}
