import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '마이페이지',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ProfileScreen(),
    );
  }
}

// 마이페이지 (ProfileScreen)
class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    String username = '홍길동';
    String email = 'example@example.com';

    return Scaffold(
      appBar: AppBar(title: Text('마이페이지')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            Text(
              '$username 님 안녕하세요!',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              '메일 주소: $email',
              style: TextStyle(
                fontSize: 18,
                color: Colors.grey,
              ),
            ),
            SizedBox(height: 30),
            _buildMenuItem(context, '내 계정', Icons.account_circle, AccountScreen()),
            _buildMenuItem(context, '고객센터', Icons.support_agent, SupportScreen()),
            _buildMenuItem(context, '자주 묻는 질문', Icons.question_answer, FAQScreen()),
            _buildMenuItem(context, '공지사항', Icons.announcement, NoticesScreen()),
            _buildMenuItem(context, '문의하기', Icons.contact_mail, ContactScreen()),
            _buildMenuItem(context, '설정', Icons.settings, SettingsScreen()),
          ],
        ),
      ),
    );
  }

  Widget _buildMenuItem(BuildContext context, String title, IconData icon, Widget screen) {
    return Column(
      children: [
        ListTile(
          title: Text(title),
          trailing: Icon(Icons.arrow_forward_ios),
          leading: Icon(icon),
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => screen),
          ),
        ),
        Divider(),
      ],
    );
  }
}

// 내 계정 화면
class AccountScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('내 계정')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('사용자 정보', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            SizedBox(height: 16),
            Text('이름: 홍길동'),
            Text('이메일: example@example.com'),
            SizedBox(height: 30),
            Center(
              child: ElevatedButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('수정하기'),
                      content: TextField(
                        decoration: InputDecoration(hintText: '새로운 이름 입력'),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: Text('취소'),
                        ),
                        TextButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          child: Text('저장'),
                        ),
                      ],
                    ),
                  );
                },
                child: Text('정보 수정'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 고객센터 화면
class SupportScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('고객센터')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text('고객센터 안내', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            Text('1. 문의처: support@example.com'),
            Text('2. 전화: 123-4567-890'),
            SizedBox(height: 30),
            Center(
              child: ElevatedButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('문의하기'),
                      content: TextField(
                        decoration: InputDecoration(hintText: '문의 내용을 입력하세요'),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: Text('취소'),
                        ),
                        TextButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          child: Text('전송'),
                        ),
                      ],
                    ),
                  );
                },
                child: Text('문의하기'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 자주 묻는 질문 화면
class FAQScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('자주 묻는 질문')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            ListTile(
              title: Text('Q1: 이 앱은 무엇인가요?'),
              subtitle: Text('A: 이 앱은 사회초년생을 위한 다양한 정보를 제공합니다.'),
            ),
            ListTile(
              title: Text('Q2: 계정 정보를 어떻게 수정하나요?'),
              subtitle: Text('A: 마이페이지에서 "내 계정"을 클릭하여 수정할 수 있습니다.'),
            ),
          ],
        ),
      ),
    );
  }
}

// 공지사항 화면
class NoticesScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('공지사항')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            ListTile(title: Text('공지사항 제목 1'), subtitle: Text('공지사항 내용 1...')),
            ListTile(title: Text('공지사항 제목 2'), subtitle: Text('공지사항 내용 2...')),
          ],
        ),
      ),
    );
  }
}

// 문의하기 화면
class ContactScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('문의하기')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('문의할 내용을 작성해 주세요.', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            TextField(
              decoration: InputDecoration(
                hintText: '문의 내용을 입력하세요',
                border: OutlineInputBorder(),
              ),
              maxLines: 5,
            ),
            SizedBox(height: 20),
            Center(
              child: ElevatedButton(
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('문의 내용 전송'),
                      content: Text('문의 내용이 전송되었습니다.'),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: Text('확인'),
                        ),
                      ],
                    ),
                  );
                },
                child: Text('문의하기'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 설정 화면
class SettingsScreen extends StatefulWidget {
  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool isDarkMode = false;
  bool isNotificationsEnabled = true;
  bool isAutoLoginEnabled = false;
  String selectedLanguage = '한국어';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('설정')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            Text('설정 항목', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            SwitchListTile(
              title: Text('다크모드'),
              value: isDarkMode,
              onChanged: (bool value) {
                setState(() {
                  isDarkMode = value;
                });
              },
            ),
            SwitchListTile(
              title: Text('알림 설정'),
              value: isNotificationsEnabled,
              onChanged: (bool value) {
                setState(() {
                  isNotificationsEnabled = value;
                });
              },
            ),
            SwitchListTile(
              title: Text('자동 로그인'),
              value: isAutoLoginEnabled,
              onChanged: (bool value) {
                setState(() {
                  isAutoLoginEnabled = value;
                });
              },
            ),
            Divider(),
            ListTile(
              title: Text('언어 설정'),
              trailing: DropdownButton<String>(
                value: selectedLanguage,
                onChanged: (String? newValue) {
                  setState(() {
                    selectedLanguage = newValue!;
                  });
                },
                items: ['한국어', 'English', '日本語', '中文']
                    .map<DropdownMenuItem<String>>((String value) {
                  return DropdownMenuItem<String>(
                    value: value,
                    child: Text(value),
                  );
                }).toList(),
              ),
            ),
            Divider(),
            ListTile(
              title: Text('데이터 동기화'),
              trailing: ElevatedButton(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('데이터를 동기화했습니다.')),
                  );
                },
                child: Text('동기화'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}