from vision.tag import Tag
from typing import List, Tuple, Dict
import math, statistics


class Box: 
    def __init__(self, knownTags: List, detectedTags: List[Tag], image=None, worldCoord=None):
        self.knownTags = knownTags
        self.detectedTags = detectedTags
        self.image = image
        self.worldCoord = worldCoord

    def calcWorldCoord(self, history): 
        """find center of box based on 4 april tags"""
        if len(self.detectedTags) == len(self.knownTags): # all tags detected
            results = [tag.calcWorldPos() for tag in self.detectedTags if tag.calcWorldPos()[0]]
            
    
            positions = [pos for _, pos, _ in results]
            thetas = [theta for _, _, theta in results]

            # Transpose positions into separate x, y, z lists
            xs, ys, zs = zip(*positions)
            worldCoord = [
                statistics.mean(xs),
                statistics.mean(ys),
                statistics.mean(zs),
                statistics.mean(thetas)
            ]

        elif history:
            
        else:
            print(f'not enough info to get position of box')
        


        self.worldCoord = worldCoord
        return worldCoord