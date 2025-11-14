# Git Flow Guide  

Pedestrian-Tracking-for-Accident-Prevention  
협업을 위한 Git Flow 규칙

---

## 1. Branch Strategy

본 프로젝트는 다음 3가지 주요 브랜치를 사용한다:

```
main        # 안정된 버전
develop     # 기능 통합 브랜치
feature/*   # 기능 개발 브랜치
```

### ✔ main

- 안정된 코드 유지  
- 직접 push 금지  
- PR 승인 후 merge  

### ✔ develop

- 기능 통합 브랜치  
- 항상 최신 develop에서 브랜치 생성  

### ✔ feature/*

- 기능 단위 개발 브랜치  
- 작업 완료 후 PR 생성 → develop 병합  
- merge 후 feature 브랜치 삭제  

---

## 2. Workflow

1. develop 최신 코드 pull  
2. feature/<task-name> 브랜치 생성  
3. 작업 후 commit  
4. feature 브랜치 push  
5. GitHub PR 생성 (base: develop)  
6. 리뷰 승인 후 merge  

---

## 3. Commit Rules (Conventional Commits)

- **feat**: 새로운 기능 추가  
- **fix**: 버그 수정  
- **chore**: 환경 설정/폴더 구성  
- **docs**: 문서 변경  
- **refactor**: 코드 리팩토링  
- **style**: 스타일/포맷 변경  
- **test**: 테스트 코드  

예시:

```
feat: integrate YOLO + DeepSORT pipeline
fix: correct foot point extraction bug
chore: initialize project structure and .gitignore
```

---

## 4. PR Rules

- 모든 작업은 PR로 merge  
- 제목 명확하게 작성  
- PR 본문: What / Why / How  
- 팀원 1명 이상 승인  
- 충돌(conflict)은 feature 브랜치에서 해결  

---

## 5. Branch Cleanup

- feature 브랜치는 merge 후 즉시 삭제  
- 오래된 브랜치는 주기적으로 정리  

---

## 6. Git Flow Summary

```
main: 안정된 버전
develop: 기능 통합
feature/*: 기능 개발

1) develop에서 브랜치 생성
2) feature에서 작업
3) PR → develop merge
4) develop 안정화 후 main 병합
```

---

