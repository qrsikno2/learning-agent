import agentscope, os
import numpy as np
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.pipeline import MsgHub
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.pipeline import fanout_pipeline
from agentscope.message import Msg
from pydantic import BaseModel, Field
from typing import List
from collections import Counter

load_dotenv()  
MAX_DISCUSSION_ROUND = 3

def format_player_list(players):
    return "[" + ", ".join(players) + "]"

class DiscussionModelCN(BaseModel):
    reach_agreement: bool = Field(description="是否达成一致意见", default=False)
    confidence_level: int = Field(description="达成一致意见的置信度,范围1-10", ge=1, le=10, default=5)
    key_evidence: str = Field(description="达成一致意见的关键证据", default=None)
    
class KillModelCN(BaseModel):
    kill_target: str = Field(description="选择的击杀/淘汰目标", default=None)
    
# class WitchActionModelCN(BaseModel):
#     use_antidote: bool = Field(description="是否使用解药", default=False)
#     use_poison: bool = Field(description="是否使用毒药", default=False)
#     target_name: str = Field(description="使用解药或毒药的目标玩家", default=None) 

# class SeerActionModelCN(BaseModel):
#     check_player: str = Field(description="预言家选择查看的玩家", default=None)
    
class GameAgent(ReActAgent):
    def __init__(self, model, name, character, game_role):
        super().__init__(
            name=name,
            sys_prompt=get_role_action_prompt(game_role, character),
            model=model,
            formatter=OpenAIChatFormatter(), 
        )
        self.character = character
        self.game_role = game_role
    
class Moderator():
    def __init__(self, werewolves: List[GameAgent] = None, villagers: List[GameAgent] = None):
        self.werewolves = werewolves
        self.villagers = villagers
    
    def _get_alive_players(self):
        alive_players = []
        if self.werewolves:
            alive_players.extend(self.werewolves)
        if self.villagers:
            alive_players.extend(self.villagers)
        return alive_players

    async def werewolf_part(self, alive_villagers: List[GameAgent], alive_players: List[GameAgent]):
        print("[狼人行动阶段]")
        if not self.werewolves:
            print("没有狼人了，跳过狼人行动阶段")
            return None
        
        async with MsgHub(
            participants=self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.announce(
                f"狼人们，请讨论今晚的击杀目标。只用决定谁是目标，系统会自动击杀. 决定完成之后如果不需要讨论只需要说出目标的名字，不用其他内容。以下是存活的村民：{format_player_list([player.name for player in alive_villagers])}" 
            )
        ) as hub:
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)
            
            hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.announce("请选出今晚的击杀目标"),
                enable_gather=False,
                structured_model=KillModelCN,
            )
            valid_targets = []
            for msg_vote in kill_votes:
                if getattr(msg_vote, "metadata", None) and hasattr(msg_vote.metadata, "kill_target"):
                    target = msg_vote.metadata.kill_target
                elif getattr(msg_vote, "metadata", None) and isinstance(msg_vote.metadata, dict):
                    # 有些大模型解析出的 dict
                    target = msg_vote.metadata.get("kill_target")
                else:
                    target = None
                    
                if target in [player.name for player in alive_players]:
                    print(f"{target} 1票")
                    valid_targets.append(target)
            
            if not valid_targets:
                return None

            vote_counter = Counter(valid_targets)
            result = vote_counter.most_common()
            print("今晚的击杀投票结果：", result)
            most_voted_target, count = result[0]
            
            return most_voted_target
    
    async def villager_part(self, alive_players: List[GameAgent], removed_player: str = None):
        print("[白天行动阶段]")
        if not self.villagers:
            print("没有村民了，跳过村民行动阶段")
            return None
        
        if removed_player:
            print(f"昨晚被击杀的玩家是{removed_player}")
        else:
            print("昨晚没有玩家被击杀")
        
        async with MsgHub(
            participants=self.villagers,
            enable_auto_broadcast=True,
            announcement=await self.announce(
                f"村民们，请讨论今天的投票目标，关于你怀疑谁最有可能是狼人。只需要投票，投票完成后系统会自动击杀。决定完成之后如果不需要讨论只需要说出目标的名字，不用其他内容。存活玩家：{format_player_list([player.name for player in alive_players])}" + (f"，昨晚被击杀的玩家是{removed_player}" if removed_player else "")
            )
        ) as hub:
            for _ in range(MAX_DISCUSSION_ROUND):
                for villager in self.villagers:
                    await villager(structured_model=DiscussionModelCN)
            
            hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.villagers,
                msg=await self.announce("请选出今天的最有可能是狼人的目标"),
                enable_gather=False,
                structured_model=KillModelCN,
            )
            
            valid_targets = []
            for vote in kill_votes:
                if isinstance(vote, dict) and "kill_target" in vote and vote["kill_target"] in [player.name for player in alive_players]:
                    valid_targets.append(vote.kill_target)
            
            if not valid_targets:
                return None

            vote_counter = Counter(valid_targets)
            most_voted_target, count = vote_counter.most_common(1)[0]
            
            return most_voted_target
            
    async def announce(self, message: str):
        return Msg(name="主持人", content=message, role="assistant")
    
    def _remove_player(self, player_name: str):
        for player in self.werewolves:
            if player.name == player_name:
                self.werewolves.remove(player)
                return

        for player in self.villagers:
            if player.name == player_name:
                self.villagers.remove(player)
                return
    
    async def run_game(self):
        while len(self.werewolves) > 0 and len(self.villagers) > 0:
            alive_players = self._get_alive_players()
            alive_villagers = [player for player in alive_players if player in self.villagers]
            kill_target_wolf = await self.werewolf_part(alive_villagers, alive_players)
            if kill_target_wolf:
                self._remove_player(kill_target_wolf)
            
            alive_players = self._get_alive_players()
            vote_target = await self.villager_part(alive_players, removed_player=kill_target_wolf)
            if vote_target:
                self._remove_player(vote_target)
        
def get_role_action_prompt(role: str, character: str) -> str:
    base_prompt = f"""你是{character}, 在这场三国狼人杀中扮演{role}的角色。
    重要规则：
    0. 游戏分为白天和夜晚两个阶段，交替进行. 夜晚狼人投票决定击杀目标，白天所有玩家投票决定谁是可疑的狼人。
    1. 你只能通过对话和推理参与游戏
    2. 不要尝试调用任何外部工具或函数
    3. 严格按照要求的JSON格式回复 
    4. 游戏中你只能看到自己的角色信息，其他玩家的身份未知。
    
    角色特点：
    """
    
    if role == "狼人":
        return base_prompt + f"""
        - 你是狼人阵营，目标是消灭所有好人
        - 夜晚可以与其他狼人协商击杀目标
        - 白天要隐藏身份，误导好人,保护同伴
        - 以{character}的性格说话和执行三国狼人杀的行动
        """

    if role == "村民":
        return base_prompt + f"""
        - 你是村民，目标是找出所有狼人并保护好人
        - 你没有特殊能力，主要通过白天的讨论和投票来判定狼人身份并击杀之
        - 白天要分析信息，揭露狼人
        - 以{character}的性格说话和执行三国狼人杀的行动
        """
        
    # if role == "预言家":
    #     return base_prompt + f"""
    #     - 你是预言家，目标是找出所有狼人并保护好人
    #     - 夜晚可以查看一个玩家的身份
    #     - 白天要分析信息，揭露狼人
    #     - 以{character}的性格说话和执行三国狼人杀的行动
    #     """
    
    # if role == "女巫":
    #     return base_prompt + f"""
    #     - 你是女巫，目标是保护好人并消灭狼人
    #     - 夜晚可以选择使用解药救人或毒药杀人
    #     - 毒药和解药只能使用一次，且不能同时使用，要谨慎选择
    #     - 白天要分析信息，做出决策
    #     - 以{character}的性格说话和执行三国狼人杀的行动
    #     """

async def game_main():
    p = 0.3

    agentscope.init(logging_path="game.log")
    llm = OpenAIChatModel(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"), 
        client_kwargs={"base_url": os.getenv("LLM_BASE_URL")},
        stream=False,
    )
    characters_pronom = ["刘备", "关羽", "张飞", "曹操", "孙权", "诸葛亮", "司马懿", "周瑜", "吕布", "赵云"]
    identity = np.random.choice(["狼人", "村民"], size=len(characters_pronom), p=[p, 1-p])
    werewolves = []
    villagers = []
    for character, role in zip(characters_pronom, identity):
        character_agent = GameAgent(model=llm, name=character, character=character, game_role=role)
        if role == "狼人":
            werewolves.append(character_agent)
        else:
            villagers.append(character_agent)
    moderator = Moderator(werewolves=werewolves, villagers=villagers)
    await moderator.run_game()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(game_main())
