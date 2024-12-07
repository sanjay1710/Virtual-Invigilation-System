Project Overview

The Virtual Invigilator project aims to develop a system that monitors students during online exams, ensuring a secure and efficient environment for conducting assessments. This system uses **computer vision (CV2)** and the **YOLO object detection algorithm** to track students' actions and identify objects in the exam environment. The goal is to detect any unauthorized activities or objects during the exam and alert both the student and invigilator in real-time. 

This Virtual Invigilator can help improve the integrity of online exams by preventing cheating and unauthorized assistance, ensuring a fair testing environment for all students.

Libraries Used

CV2 (OpenCV): A library used for video capture, real-time image processing, and object detection.
pyttsx3: A text-to-speech library that allows the system to give voice-based alerts and notifications.
OS: Provides functionality for interacting with the operating system to manage files and processes.
Argparse: A Python library used to handle command-line arguments and make the system configurable from the terminal.
YOLO (You Only Look Once): A real-time object detection algorithm used to identify objects such as mobile phones, books, or other prohibited items in the student's environment.

Features

Real-Time Monitoring: The system captures video in real-time to monitor the student’s actions during the exam.
Object Detection: Utilizes the YOLO algorithm to detect various objects in the exam environment (e.g., mobile phones, books, or any unauthorized items).
Speech Alerts: Uses pyttsx3 to alert the student or invigilator if any unauthorized object is detected, ensuring that the invigilator is notified immediately.
Customizable via Command-Line: The system can be configured using command-line arguments with argparse, allowing users to specify video sources, YOLO configurations, and other parameters.
Efficient and Secure: Helps maintain the integrity of online assessments by automatically detecting suspicious activities or cheating attempts.

Breakdown of Sections:

1. Project Overview: Describes the objective of the Virtual Invigilator system, focusing on secure online exams and object detection.
2. Libraries Used: Lists the Python libraries and technologies used, such as OpenCV, YOLO, pyttsx3, and argparse.
3. Features: Explains the core functionality of the system, including real-time monitoring, object detection, and speech-based alerts.
4. Installation: Provides step-by-step instructions on how to install the necessary libraries and run the system.
5. Running the System: Describes how to execute the system with command-line arguments for configuration and customization.
6. Example Output: Shows what the user can expect when running the system, including real-time object detection and alerts.
7. Findings and Insights: Highlights the effectiveness of the system in detecting objects and providing alerts.
8. Future Improvements: Suggests potential enhancements, such as upgrading YOLO or adding facial recognition.
