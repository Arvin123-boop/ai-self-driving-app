from flask import Flask, render_template, Response
from road_env import RoadEnv
from dqn_agent import DQNAgent
import cv2

app = Flask(__name__)
env = RoadEnv()
agent = DQNAgent()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size=32)
        state = next_state

        frame = env.get_frame()
        cv2.putText(frame, f"Score: {env.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f"High Score: {env.high_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
