import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 포트폴리오 관련 질문과 답변 데이터
questions = [
    "주제가 뭐예요?",
    "어떤 모델을 사용했나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "맡은 업무는 무엇인가요?",
    "조원은 총 몇 명인가요?",
    "어떤 데이터를 사용했나요?",
    "어떤 점이 힘들었나요?",
    "조장은 누구예요?",
    "무엇을 배웠나요?",
]

answers = [
    "딥러닝을 이용한 무인 점포 절도 탐지입니다.",
    "YOLOV8과 Mediapipe를 사용했습니다.",
    "10/28~11/17 대략 3주 동안입니다.",
    "공통 업무로 데이터 전처리, 라벨링이 있고 저는 모델 구축을 맡았습니다.",
    "총 4명이서 진행했습니다.",
    "AI Hub에서 무인점포 이상 행동 데이터, 그리고 유튜브에서 직접 영상을 구했습니다.",
    "스켈레톤 처리를 위한 라벨링 작업과 객체 탐지를 위한 모델 구축이 힘들었으나 여러 문헌들을 참고해가며 조원들과 합심해 진행해나갔습니다.",
    "이혜인입니다.",
    "아무리 어려워 보이는 기술도 차근차근 풀어나가보면 충분히 구현할 수 있다는 점과 혼자 진행하는 것보다 팀원들과 협업했을 때 결과물이 훨씬 잘 나온다는 점도 배웠습니다. ",
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포트폴리오 챗봇")
st.write("저희가 만든 포트폴리오에 관한 질문을 입력해보세요. 예: 주제가 어떻게 되나요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")