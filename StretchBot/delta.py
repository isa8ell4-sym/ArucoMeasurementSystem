import statistics

class Delta:
    def __init__(self, currImg, prevImg, deltaX=None, deltaY=None, deltaZ=None, deltaTheta=None):
        self.currImg = currImg
        self.prevImg = prevImg
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.deltaZ = deltaZ
        self.deltaTheta = deltaTheta
    
    def getPrevTagsIDs(self):
        if self.prevImg.tags is not None:
            return [tag.id for tag in self.prevImg.tags]
        else:
            return []
    
    def getCurrTagsIDs(self):
        if self.currImg.tags is not None:
            return [tag.id for tag in self.currImg.tags]
        else:
            return []
    
    def calcDeltas(self):
        allX = []
        allY = []
        allZ = []
        allTheta = []

        prevTagIds = self.getPrevTagsIDs()
        currTagIds = self.getCurrTagsIDs()
        print(f"prev: {[int(i) for i in prevTagIds]}, curr: {[int(i) for i in currTagIds]}")

        if len(prevTagIds) > 0 and len(currTagIds) > 0:
            
            prevTags = {tag.id:tag for tag in self.prevImg.tags}
            currTags = {tag.id:tag for tag in self.currImg.tags}

            commonIds = list(set(prevTagIds) & set(currTagIds))
            print(f"commonIds: {[int(i) for i in commonIds]}")
            for id in commonIds:
                prev = prevTags[id]
                curr = currTags[id]

                allX.append(curr.worldCoord[0] - prev.worldCoord[0])
                allY.append(curr.worldCoord[1] - prev.worldCoord[1])
                allZ.append(curr.worldCoord[2] - prev.worldCoord[2])
                allTheta.append(curr.theta - prev.theta)

            deltaX = statistics.mean(allX)
            deltaY = statistics.mean(allY)
            deltaZ = statistics.mean(allZ)
            deltaTheta = statistics.mean(allTheta)

            self.deltaX = deltaX
            self.deltaY = deltaY
            self.deltaZ = deltaZ
            self.deltaTheta = deltaTheta

            print(f"overall shift between {self.currImg.name} & {self.prevImg.name} is {[int(deltaX), int(deltaY), int(deltaZ), int(deltaTheta)]}\n")

            return [deltaX, deltaY, deltaZ, deltaTheta]
        elif len(prevTagIds) > 0: # no tags detected in curr photo
            prevTags = {tag.id:tag for tag in self.prevImg.tags}
            # currTags = {tag.id:tag for tag in self.currImg.tags}
            for id in prevTagIds:
                prev = prevTags[id]
                # curr = currTags[id]

                allX.append(0 - prev.worldCoord[0])
                allY.append(0 - prev.worldCoord[1])
                allZ.append(0 - prev.worldCoord[2])
                allTheta.append(0 - prev.theta)

            deltaX = statistics.mean(allX)
            deltaY = statistics.mean(allY)
            deltaZ = statistics.mean(allZ)
            deltaTheta = statistics.mean(allTheta)

            self.deltaX = deltaX
            self.deltaY = deltaY
            self.deltaZ = deltaZ
            self.deltaTheta = deltaTheta

            return [deltaX, deltaY, deltaZ, deltaTheta]
        
        elif len(currTagIds) > 0: # no tags detected in prev photo 
            # prevTags = {tag.id:tag for tag in self.prevImg.tags}
            currTags = {tag.id:tag for tag in self.currImg.tags}
            for id in currTagIds:
                # prev = prevTags[id]
                curr = currTags[id]

                allX.append(curr.worldCoord[0])
                allY.append(curr.worldCoord[1])
                allZ.append(curr.worldCoord[2])
                allTheta.append(curr.theta)

            deltaX = statistics.mean(allX)
            deltaY = statistics.mean(allY)
            deltaZ = statistics.mean(allZ)
            deltaTheta = statistics.mean(allTheta)

            self.deltaX = deltaX
            self.deltaY = deltaY
            self.deltaZ = deltaZ
            self.deltaTheta = deltaTheta

            return [deltaX, deltaY, deltaZ, deltaTheta]
        else:
            print(f"not enough tags detected between {self.prevImg.name} and {self.currImg.name}")
            return [None, None, None, None]