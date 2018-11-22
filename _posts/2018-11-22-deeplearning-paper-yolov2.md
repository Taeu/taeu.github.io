---
layout: post
title: "[논문] YOLO9000: Better, Faster, Stronger 분석"
category: paper
tags: dl paper
comments: true
img: ssd1.jpg
---

# YOLO9000: Better, Faster, Stronger
---

![01]()

Image Detection의 Hot한 YOLO 알고리즘의 2번째 버전 **YOLO9000: Better, Faster, Stronger** 논문에 대한 분석을 해보았다. 앞서 [YOLO v1: You Only Look Once](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)과 [SSD: Single Shot Multibox Detector](https://taeu.github.io/paper/deeplearning-paper-ssd/)을 살펴봤으므로 이번 논문에서는 많은 부분들에 대한 설명들은 생략하고 깔끔하게 정리하되, 설명이 추가적으로 필요한 부분을 집중적으로 다루고자 한다. 따라서 읽다가 혹시 이해되지 않는 부분이 있다면 앞선 논문의 내용을 살펴보길 바란다.

# 0. Abstract
---

 이번 논문은 크게 3 부분으로 나뉜다. 

- Better : Accruacy , mAP 측면의 개선사항
- Faster : 속도 개선
- Stronger : 더 많은, 다양한 클래스 예측

그럼 이제 하나씩 살펴보자.


# 1. Better
---

![table2]()
- Batch normalization
-  High resolution classifier
- Convolutional with Anchor boxes
-  Dimension clusters
-  Direct location prediction
-  Fine-grained features
-  Multi-scale training




