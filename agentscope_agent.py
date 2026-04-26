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
    
class WitchActionModelCN(BaseModel):
    use_antidote: bool = Field(description="是否使用解药", default=False)
    use_poison: bool = Field(description="是否使用毒药", default=False)
    target_name: str = Field(description="使用解药或毒药的目标玩家", default=None) 

class SeerActionModelCN(BaseModel):
    check_player: str = Field(description="预言家选择查看的玩家", default=None)
    
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
    def __init__(self, werewolves: List[GameAgent] = None, villagers: List[GameAgent] = None, witch: GameAgent = None, seer: GameAgent = None):
        self.werewolves = werewolves or []
        self.villagers = villagers or []
        self.witch = witch
        self.seer = seer
        self.witch_has_antidote = True
        self.witch_has_poison = True
    
    def _get_alive_players(self):
        alive_players = []
        if self.werewolves:
            alive_players.extend(self.werewolves)
        if self.villagers:
            alive_players.extend(self.villagers)
        if self.witch:
            alive_players.append(self.witch)
        if self.seer:
            alive_players.append(self.seer)
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
                agreements = []
                for wolf in self.werewolves:
                    msg = await wolf(structured_model=DiscussionModelCN)
                    metadata = getattr(msg, "metadata", None)
                    agreed = False
                    if metadata:
                        if hasattr(metadata, "reach_agreement"):
                            agreed = metadata.reach_agreement
                        elif isinstance(metadata, dict):
                            agreed = metadata.get("reach_agreement", False)
                    agreements.append(agreed)
                
                if len(self.werewolves) > 0 and all(agreements):
                    break
            
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
        if not alive_players:
            print("没有存活玩家了，跳过白天行动阶段")
            return None
        
        if removed_player:
            print(f"昨晚被击杀的玩家是{removed_player}")
        else:
            print("昨晚没有玩家被击杀")
        
        async with MsgHub(
            participants=alive_players,
            enable_auto_broadcast=True,
            announcement=await self.announce(
                f"各位玩家，现在是白天，请大家讨论今天的投票目标，关于你怀疑谁最有可能是狼人。只需要投票，投票完成后系统会自动击杀。决定完成之后如果不需要讨论只需要说出目标的名字，不用其他内容。存活玩家：{format_player_list([player.name for player in alive_players])}" + (f"，昨晚被击杀的玩家是{removed_player}" if removed_player else "昨晚没有玩家被击杀")
            )
        ) as hub:
            for _ in range(MAX_DISCUSSION_ROUND):
                agreements = []
                for player in alive_players:
                    msg = await player(structured_model=DiscussionModelCN)
                    metadata = getattr(msg, "metadata", None)
                    agreed = False
                    if metadata:
                        if hasattr(metadata, "reach_agreement"):
                            agreed = metadata.reach_agreement
                        elif isinstance(metadata, dict):
                            agreed = metadata.get("reach_agreement", False)
                    agreements.append(agreed)
                
                if len(alive_players) > 0 and all(agreements):
                    break
            
            hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                alive_players,
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
                
        if self.witch and self.witch.name == player_name:
            self.witch = None
            return
            
        if self.seer and self.seer.name == player_name:
            self.seer = None
            return
            
    async def seer_part(self, alive_players: List[GameAgent]):
        print("[预言家行动阶段]")
        if not self.seer:
            print("预言家已死或不存在，跳过预言家行动阶段")
            return
        
        msg = await self.announce(f"预言家，请选择你要查验的玩家。存活玩家：{format_player_list([p.name for p in alive_players])}")
        response = await self.seer(msg, structured_model=SeerActionModelCN)
        
        metadata = getattr(response, "metadata", None)
        target = None
        if metadata:
            if hasattr(metadata, "check_player"):
                target = metadata.check_player
            elif isinstance(metadata, dict):
                target = metadata.get("check_player")
        
        if target and target in [p.name for p in alive_players]:
            is_werewolf = any(wolf.name == target for wolf in self.werewolves)
            identity_str = "狼人" if is_werewolf else "好人"
            result_msg = f"系统提示：你查验的玩家 {target} 的身份是【{identity_str}】。"
            await self.seer(await self.announce(result_msg))
        else:
            await self.seer(await self.announce("未选择有效目标或目标错误，放弃查验。"))

    async def witch_part(self, alive_players: List[GameAgent], wolf_kill_target: str):
        print("[女巫行动阶段]")
        if not self.witch:
            print("女巫已死或不存在，跳过女巫行动阶段")
            return wolf_kill_target, None
        
        prompt = f"女巫，今晚被狼人击杀的玩家是：{wolf_kill_target if wolf_kill_target else '没人'}。你有一瓶解药（状态：{'可用' if self.witch_has_antidote else '已用'}）和一瓶毒药（状态：{'可用' if self.witch_has_poison else '已用'}）。你可以选择用解药救人，或者用毒药毒死存活玩家：{format_player_list([p.name for p in alive_players])}。注意一晚只能用一瓶药，也可都不用。"
        msg = await self.announce(prompt)
        response = await self.witch(msg, structured_model=WitchActionModelCN)
        
        metadata = getattr(response, "metadata", None)
        use_antidote = False
        use_poison = False
        target_name = None
        
        if metadata:
            if hasattr(metadata, "use_antidote"):
                use_antidote = metadata.use_antidote
                use_poison = metadata.use_poison
                target_name = metadata.target_name
            elif isinstance(metadata, dict):
                use_antidote = metadata.get("use_antidote", False)
                use_poison = metadata.get("use_poison", False)
                target_name = metadata.get("target_name")
        
        final_kill_target = wolf_kill_target
        killed_by_witch = None
        
        if use_antidote and self.witch_has_antidote and wolf_kill_target:
            print("女巫使用了解药。")
            self.witch_has_antidote = False
            final_kill_target = None
        elif use_poison and self.witch_has_poison and target_name in [p.name for p in alive_players]:
            print(f"女巫使用了毒药，毒死了 {target_name}。")
            self.witch_has_poison = False
            killed_by_witch = target_name
            
        return final_kill_target, killed_by_witch
    
    async def run_game(self):
        while len(self.werewolves) > 0 and len(self._get_alive_players()) - len(self.werewolves) > 0:
            alive_players = self._get_alive_players()
            alive_good = [player for player in alive_players if player not in self.werewolves]
            
            # 夜晚阶段
            kill_target_wolf = await self.werewolf_part(alive_good, alive_players)
            
            alive_players = self._get_alive_players()
            final_kill_target, killed_by_witch = await self.witch_part(alive_players, kill_target_wolf)
            
            alive_players = self._get_alive_players()
            await self.seer_part(alive_players)
            
            # 结算昨晚的死者
            dead_tonight = []
            if final_kill_target and final_kill_target not in dead_tonight:
                dead_tonight.append(final_kill_target)
            if killed_by_witch and killed_by_witch not in dead_tonight:
                dead_tonight.append(killed_by_witch)
                
            for dead in dead_tonight:
                self._remove_player(dead)
            
            # 白天阶段
            alive_players = self._get_alive_players()
            if not alive_players or len(self.werewolves) == 0 or len(alive_players) - len(self.werewolves) == 0:
                print("游戏结束！")
                break
            
            dead_str = "和".join(dead_tonight) if dead_tonight else None
            vote_target = await self.villager_part(alive_players, removed_player=dead_str)
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
        
    if role == "预言家":
        return base_prompt + f"""
        - 你是预言家，目标是找出所有狼人并保护村民. 
        - 你是村民阵营的，你可以暴露自己的身份来增加好人阵营的胜率，但也可能因此成为狼人的首要击杀目标。
        - 夜晚可以查看一个玩家的身份
        - 白天要分析信息，揭露狼人
        - 以{character}的性格说话和执行三国狼人杀的行动
        """
    
    if role == "女巫":
        return base_prompt + f"""
        - 你是女巫，目标是保护好人并消灭狼人
        - 你是村民阵营的，你可以暴露自己的身份来增加好人阵营的胜率，但也可能因此成为狼人的首要击杀目标。
        - 夜晚可以选择使用解药救人或毒药杀人
        - 毒药和解药只能使用一次，且不能同时使用，要谨慎选择
        - 白天要分析信息，做出决策
        - 以{character}的性格说话和执行三国狼人杀的行动
        """

async def game_main():
    agentscope.init(logging_path="game.log")
    llm = OpenAIChatModel(
        model_name=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"), 
        client_kwargs={"base_url": os.getenv("LLM_BASE_URL")},
        stream=False,
    )
    characters_pronom = ["刘备", "关羽", "张飞", "曹操", "孙权", "诸葛亮", "司马懿", "周瑜", "吕布", "赵云"]
    
    roles = ["狼人"] * 3 + ["女巫"] * 1 + ["预言家"] * 1 + ["村民"] * 5
    identity = np.random.permutation(roles)
    
    werewolves = []
    villagers = []
    witch = None
    seer = None
    
    for character, role in zip(characters_pronom, identity):
        character_agent = GameAgent(model=llm, name=character, character=character, game_role=role)
        if role == "狼人":
            werewolves.append(character_agent)
        elif role == "女巫":
            witch = character_agent
        elif role == "预言家":
            seer = character_agent
        else:
            villagers.append(character_agent)
            
    moderator = Moderator(werewolves=werewolves, villagers=villagers, witch=witch, seer=seer)
    await moderator.run_game()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(game_main())
