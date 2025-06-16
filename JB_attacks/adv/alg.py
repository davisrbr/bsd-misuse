import torch

class GWW():
    def __init__(self, num_prompts):
        self.num_prompts= num_prompts
        self.content = []
    
    def add_prompt(self, init_prompt, losses, messages):
        self.content.append({"init_prompt": init_prompt, 
                                "mean_loss": torch.mean(losses),
                                "losses": losses,
                                "messages": messages})
        
        self.sort_prompts()
        
        if len(self.content) > self.num_prompts:
            self.content = self.content[:self.num_prompts]
    
    def sort_prompts(self):
        self.content = sorted(self.content, key=lambda x: x["mean_loss"])
        
    def get_prompt(self):
        return self.content[0]["init_prompt"], self.content[0]["losses"], self.content[0]["messages"]
    

class GWW_dfs_min(GWW):
    def sort_prompts(self):
        self.content = sorted(self.content, key=lambda x: torch.min(x["losses"]))
        
    def get_prompt(self):
        first_content = self.content[0]
        self.content.pop(0)
        
        return first_content["init_prompt"], first_content["losses"], first_content["messages"]