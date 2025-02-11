import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'diary.dart'; // diary.dart 파일을 import
import 'heart.dart'; // heart.dart 파일을 import

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String weatherInfo = "로딩 중...";
  String cityName = "Seoul"; // 도시 이름
  String quote = "“할 수 있다면, 해야 한다.”"; // 명언 기본값
  List<Map<String, dynamic>> tasks = [
    {"title": "업무 준비하기", "isChecked": false},
    {"title": "은행 가기", "isChecked": false},
  ];
  String newsHeadline = "오늘의 추천 뉴스 로딩 중..."; // 오늘의 뉴스 제목
  String newsImageUrl = ""; // 뉴스 이미지 URL
  String newTaskTitle = ""; // 할 일 입력 값을 저장하는 변수
  List<String> events = []; // 일정을 저장할 변수

  Future<void> getWeather() async {
    final String apiKey = '@@@'; // API 키
    final String url =
        'http://api.weatherapi.com/v1/current.json?key=$apiKey&q=${Uri.encodeComponent(cityName)}&lang=ko'; // 도시 이름 인코딩

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(utf8.decode(response.bodyBytes));
        setState(() {
          weatherInfo =
          "오늘 날씨: ${data['current']['condition']['text']}, 기온: ${data['current']['temp_c']}°C";
        });
      } else {
        setState(() {
          weatherInfo = "날씨 정보를 가져오는 데 실패했습니다.";
        });
      }
    } catch (e) {
      setState(() {
        weatherInfo = "에러 발생: $e";
      });
    }
  }

  Future<void> getNews() async {
    final String apiKey = '@@@'; // News API 키
    final String url =
        'https://newsapi.org/v2/top-headlines?country=us&category=general&apiKey=$apiKey';

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['articles'] != null && data['articles'].isNotEmpty) {
          setState(() {
            newsHeadline = data['articles'][0]['title'] ?? "추천 뉴스 없음";
            newsImageUrl = data['articles'][0]['urlToImage'] ?? "";
          });
        } else {
          setState(() {
            newsHeadline = "추천 뉴스가 없습니다.";
            newsImageUrl = "";
          });
        }
      } else {
        setState(() {
          newsHeadline = "뉴스 정보를 가져오는 데 실패했습니다.";
          newsImageUrl = "";
        });
      }
    } catch (e) {
      setState(() {
        newsHeadline = "에러 발생: $e";
        newsImageUrl = "";
      });
    }
  }

  @override
  void initState() {
    super.initState();
    getWeather();
    getNews();
  }

  // DiaryScreen에서 일정 데이터를 받아오는 함수
  void _updateEvents(List<String> newEvents) {
    setState(() {
      events = newEvents;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            GestureDetector(
              onTap: () {
                // foot.png 아이콘을 클릭하면 diary.dart로 이동
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => DiaryScreen(updateEventsCallback: _updateEvents),
                  ),
                );
              },
              child: Image.asset(
                'assets/foot.png',
                color: Colors.grey,
                width: 24,
                height: 24,
              ),
            ),
            Text("홈"),
            GestureDetector(
              onTap: () {
                // 하트 아이콘을 클릭하면 heart.dart로 이동
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => HeartScreen()),  // HeartScreen으로 이동
                );
              },
              child: Icon(Icons.favorite, color: Colors.grey),
            ),
          ],
        ),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                "[홍길동]님! 오늘도 화이팅!",
                style: TextStyle(
                    fontSize: 24, fontWeight: FontWeight.bold, color: Colors.black),
              ),
              SizedBox(height: 16),
              Row(
                children: [
                  Icon(Icons.wb_sunny, size: 40, color: Colors.orange),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      weatherInfo,
                      style: TextStyle(fontSize: 18, color: Colors.black),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 16),
              Divider(), // 구분선 추가
              Text(
                "오늘의 할 일:",
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
              ),
              SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      onChanged: (value) {
                        newTaskTitle = value;
                      },
                      decoration: InputDecoration(
                        hintText: "할 일을 입력하세요...",
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8.0),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  ElevatedButton(
                    onPressed: () {
                      if (newTaskTitle.trim().isNotEmpty) {
                        setState(() {
                          tasks.add({
                            "title": newTaskTitle.trim(),
                            "isChecked": false,
                          });
                          newTaskTitle = ""; // 입력값 초기화
                        });
                      }
                    },
                    child: Text("추가"),
                  ),
                ],
              ),
              SizedBox(height: 16),
              ListView.builder(
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                itemCount: tasks.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(
                      tasks[index]["title"],
                      style: TextStyle(
                        color: Colors.black,
                        decoration: tasks[index]["isChecked"]
                            ? TextDecoration.lineThrough
                            : null,
                      ),
                    ),
                    trailing: Checkbox(
                      value: tasks[index]["isChecked"],
                      onChanged: (bool? value) {
                        setState(() {
                          tasks[index]["isChecked"] = value ?? false;
                        });
                      },
                    ),
                  );
                },
              ),
              SizedBox(height: 16),
              Divider(), // 구분선 추가
              Text(
                "오늘의 일정:",
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
              ),
              SizedBox(height: 8),
              events.isEmpty
                  ? Text("오늘의 일정이 없습니다.")
                  : ListView.builder(
                shrinkWrap: true,
                itemCount: events.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(events[index]),
                  );
                },
              ),
              SizedBox(height: 16),
              Divider(), // 구분선 추가
              Text(
                "오늘의 추천 뉴스:",
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
              ),
              SizedBox(height: 8),
              newsImageUrl.isNotEmpty
                  ? Image.network(newsImageUrl)
                  : Container(),
              SizedBox(height: 8),
              Card(
                elevation: 4,
                child: ListTile(
                  title: Text(newsHeadline),
                  onTap: () {},
                ),
              ),
              SizedBox(height: 16),
              Divider(), // 구분선 추가
              Text(
                "오늘의 명언:",
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
              ),
              SizedBox(height: 8),
              Text(
                quote,
                style: TextStyle(
                    fontSize: 18,
                    fontStyle: FontStyle.italic,
                    color: Colors.black),
              ),
              SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
