import 'package:flutter/material.dart';
import 'package:table_calendar/table_calendar.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class DiaryScreen extends StatefulWidget {
  final Function(List<String>) updateEventsCallback;

  DiaryScreen({required this.updateEventsCallback});

  @override
  _DiaryScreenState createState() => _DiaryScreenState();
}

class _DiaryScreenState extends State<DiaryScreen> {
  DateTime _selectedDate = DateTime.now();
  Map<DateTime, List<String>> _events = {};
  TextEditingController _controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadEvents();
  }

  _loadEvents() async {
    final prefs = await SharedPreferences.getInstance();
    final eventsString = prefs.getString('events');
    if (eventsString != null) {
      final Map<String, dynamic> eventsMap = jsonDecode(eventsString);
      setState(() {
        _events = eventsMap.map((key, value) {
          final date = DateTime.parse(key);
          final eventList = List<String>.from(value);
          return MapEntry(date, eventList);
        });
      });
    }
  }

  _saveEvents() async {
    final prefs = await SharedPreferences.getInstance();
    final eventsString = jsonEncode(
      _events.map((key, value) {
        return MapEntry(key.toIso8601String(), value);
      }),
    );
    await prefs.setString('events', eventsString);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Image.asset(
              'assets/foot.png',  // 아이콘 이미지 경로
              height: 30,  // 아이콘 높이
              width: 30,   // 아이콘 너비
            ),
            SizedBox(width: 8), // 텍스트와 아이콘 사이 간격
            Text('일기장'),
          ],
        ),
        backgroundColor: Color(0xFFFFE4B5), // AppBar 배경색 변경
      ),
      body: SingleChildScrollView( // 추가된 부분
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // 캘린더 위젯
              TableCalendar(
                firstDay: DateTime(2020),
                lastDay: DateTime(2030),
                focusedDay: _selectedDate,
                selectedDayPredicate: (day) => isSameDay(_selectedDate, day),
                onDaySelected: (selectedDay, focusedDay) {
                  setState(() {
                    _selectedDate = selectedDay;
                  });
                },
              ),
              SizedBox(height: 20),

              // 일정 추가 필드
              TextField(
                controller: _controller,
                decoration: InputDecoration(
                  labelText: '오늘의 일정을 추가하세요.',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 10),
              ElevatedButton(
                onPressed: _addEvent,
                child: Text('일정 추가'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFFFFE4B5),
                ),
              ),
              SizedBox(height: 20),

              // 선택된 날짜의 일정 표시
              _events[_selectedDate]?.isEmpty ?? true
                  ? Center(child: Text('오늘의 일정이 없습니다.'))
                  : ListView(
                shrinkWrap: true, // ListView가 남는 공간만 사용하게 설정
                children: _events[_selectedDate]!
                    .map((event) => ListTile(
                  title: Text(event),
                  trailing: IconButton(
                    icon: Icon(Icons.delete),
                    onPressed: () {
                      _deleteEvent(event);
                    },
                  ),
                ))
                    .toList(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _addEvent() {
    if (_controller.text.isEmpty) return;
    setState(() {
      if (_events[_selectedDate] == null) {
        _events[_selectedDate] = [];
      }
      _events[_selectedDate]!.add(_controller.text);
      widget.updateEventsCallback(_events[_selectedDate]!);
      _saveEvents();
      _controller.clear();
    });
  }

  void _deleteEvent(String event) {
    setState(() {
      _events[_selectedDate]?.remove(event);
      _saveEvents();
    });
  }
}
