import gradio as gr
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
import random
import json

# COMPLETE ENVIRONMENT (No external files needed)
class Email(BaseModel):
    id: str
    subject: str
    sender: str
    labels: List[str]

class Action(BaseModel):
    type: str
    email_id: str
    params: dict

class Observation(BaseModel):
    inbox: List[Email]
    current_email: Email
    score: float
    done: bool

class EmailTriageEnv:
    def __init__(self):
        self.reset()
    
    def reset(self) -> Observation:
        emails = [
            Email(id="e0", subject="WIN $1000 FREE!!!", sender="spam@lottery.com", labels=["spam"]),
            Email(id="e1", subject="URGENT: Account suspended", sender="security@fake.com", labels=["spam"]),
            Email(id="e2", subject="Team meeting 2PM", sender="manager@company.com", labels=["work"]),
            Email(id="e3", subject="Q1 Financial Report", sender="finance@company.com", labels=["work"]),
            Email(id="e4", subject="Password reset help", sender="user@company.com", labels=["support"])
        ]
        random.shuffle(emails)
        self.emails = emails
        self.current_idx = 0
        self.processed = []
        self.score = 0.0
        return self.get_obs()
    
    def step(self, action_dict: dict) -> dict:
        action = Action(**action_dict)
        self.processed.append(action.dict())
        
        # Reward calculation
        email = next((e for e in self.emails if e.id == action.email_id), None)
        pred_labels = action.params.get("labels", [])
        correct = any(l in pred_labels for l in email.labels) if email else False
        reward = 1.0 if correct else -0.5
        
        self.current_idx += 1
        self.score = min(1.0, len([p for p in self.processed if self.is_correct(p)]) / max(1, len(self.processed)))
        done = self.current_idx >= len(self.emails)
        
        return {
            "observation": self.get_obs().dict(),
            "reward": reward,
            "done": done,
            "score": self.score
        }
    
    def get_obs(self) -> Observation:
        current = self.emails[self.current_idx] if self.current_idx < len(self.emails) else self.emails[0]
        return Observation(
            inbox=self.emails[:3],
            current_email=current,
            score=self.score,
            done=self.current_idx >= len(self.emails)
        )
    
    def is_correct(self, action: dict) -> bool:
        email_id = action["email_id"]
        pred = action["params"].get("labels", [])
        email = next((e for e in self.emails if e.id == email_id), None)
        return email and any(l in pred for l in email.labels)

# HF GRADIO INTERFACE
env = EmailTriageEnv()

def reset_env():
    obs = env.reset()
    return format_obs(obs), "✅ Reset complete! Classify emails.", 0.0

def take_step(action_type, labels):
    if not action_type or not labels:
        return format_obs(env.get_obs()), "❌ Select action and labels", env.score
    
    # Find current email
    current_email = env.emails[env.current_idx] if env.current_idx < len(env.emails) else env.emails[0]
    
    action = {
        "type": action_type,
        "email_id": current_email.id,
        "params": {"labels": labels.split(",")}
    }
    
    result = env.step(action)
    obs = result["observation"]
    score = result["score"]
    
    return format_obs(obs), f"✅ Action taken! Score: {score:.2f}", score

def format_obs(obs: Observation) -> str:
    lines = ["📧 INBOX:"]
    for email in obs.inbox:
        lines.append(f"• {email.subject} ({email.sender})")
    lines.append(f"\n🎯 Current: {obs.current_email.subject}")
    lines.append(f"📊 Score: {obs.score:.2f}")
    lines.append(f"✅ Done: {obs.done}")
    return "\n".join(lines)

# GRADIO UI
with gr.Blocks(title="📧 Email Triage OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 Email Triage Assistant\nReal-world customer support task")
    
    with gr.Row():
        with gr.Column(scale=2):
            obs_output = gr.Textbox(label="Environment State", lines=12)
            score_display = gr.Number(label="Task Score (0-1)", precision=2)
        
        with gr.Column(scale=1):
            action_type = gr.Dropdown(
                choices=["classify", "archive"], 
                value="classify",
                label="Action"
            )
            labels_input = gr.Textbox(
                placeholder="spam, work, support (comma separated)",
                label="Labels"
            )
            step_btn = gr.Button("🚀 Take Step", variant="primary")
    
    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
    
    # Events
    reset_btn.click(reset_env, outputs=[obs_output, gr.Textbox(), score_display])
    step_btn.click(take_step, inputs=[action_type, labels_input], 
                   outputs=[obs_output, gr.Textbox(), score_display])
    
    gr.Markdown("""
## 🎯 Tasks
- **Easy**: Classify 5 emails (spam/work/support)
- **Score**: 1.0 = Perfect classification
- **Reward**: +1 correct, -0.5 wrong

**Try:** "classify" + "spam" for lottery emails!
    """)

if __name__ == "__main__":
    demo.launch()