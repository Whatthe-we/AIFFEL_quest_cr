import 'package:flutter/material.dart';

class HeartScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('푸시 알림 목록'),
      ),
      body: ListView(
        padding: EdgeInsets.all(16.0),
        children: [
          // 푸시 알림 1
          _buildNotificationCard(
            context,
            '이벤트 알림: 오늘의 특별 할인! 최대 50% 할인받으세요!',
            Icons.notifications_active,
          ),
          // 푸시 알림 2
          _buildNotificationCard(
            context,
            '광고 알림: 새로 출시된 제품을 확인하세요!',
            Icons.local_offer,
          ),
          // 푸시 알림 3
          _buildNotificationCard(
            context,
            '업데이트 알림: 새로운 기능이 추가되었습니다!',
            Icons.update,
          ),
          // 푸시 알림 4
          _buildNotificationCard(
            context,
            '시스템 알림: 서버 점검이 완료되었습니다.',
            Icons.settings,
          ),
        ],
      ),
    );
  }

  // 푸시 알림 카드 위젯
  Widget _buildNotificationCard(BuildContext context, String message, IconData icon) {
    return Card(
      margin: EdgeInsets.only(bottom: 12.0),
      elevation: 5.0,
      child: ListTile(
        leading: Icon(icon, color: Colors.blue, size: 40),
        title: Text(
          message,
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        subtitle: Text(
          '알림 받은 시간: 1분 전',
          style: TextStyle(fontSize: 12, color: Colors.grey),
        ),
        trailing: Icon(Icons.more_horiz, color: Colors.grey),
      ),
    );
  }
}
