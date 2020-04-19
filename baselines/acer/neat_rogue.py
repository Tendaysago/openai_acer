import neat
import numpy as np
import types
from . import run_neat_base
import random
import math
from time import sleep
import copy
from rogueinabox_lib.parser import RogueParser
RB = None
RBParser = None
FrameInfo = None
dx = [ 0, 1, 0, -1]
dy = [ -1, 0, 1, 0]
command =['k','l','j','h']
RoomInfoList = []
maxRoomID = -1
VisitedRoom = None
StairsFound = False
ExploreStack = False
class RoomInfo:
    def __init__(self,id,leftup,rightbottom,doorlist,knowndoorlist,stairexisted,stairsRoomdis):
        self.id = id
        self.leftup = leftup
        self.rightbottom = rightbottom
        self.doorlist = doorlist
        self.knowndoorlist = knowndoorlist
        self.stairexisted = stairexisted
        self.stairsRoomdis = starisRoomdis

class DoorInfo:
    def __init__(self,id,Y,X,visited,passagelist):
        self.id = id
        self.Y = Y
        self.X = X
        self.visited = False
        self.passagelist = passagelist

class PassageInfo:
    def __init__(self,id,passage,connectroomid):
        self.id = id
        self.passage = passage
        self.connectroomid = connectroomid

def MakeInput(nowRoomID):
    input = numpy.zeros(22)
    if(StairsFound):
        inputs[0]=1.0
    
    Room = RoomInfoList[nowRoomID]

    for d in range(len(Room.doorlist)):
        

def RoominfoPrint(nowRoomID):
    print("RoomInformation print!")
    print("Now Room ID: {0}".format(nowRoomID))
    print("LeftUp Y {0}, X{1},".format(RoomInfoList[nowRoomID].leftup[0],RoomInfoList[nowRoomID].leftup[1]))
    print("RightBottom Y {0}, X{1},".format(RoomInfoList[nowRoomID].rightbottom[0],RoomInfoList[nowRoomID].rightbottom[1]))
    Room = RoomInfoList[nowRoomID]
    for d in range(len(Room.doorlist)):
        door = Room.doorlist[d]
        print("Door ID:{0}, Coordinates:({1},{2}), Visited:{3},.".format(door.id,door.Y,door.X,door.visited))
        for p in range(len(door.passagelist)):
            Passage = door.passagelist[p]
            route = copy.deepcopy(Passage.passage)
            for tmp in range(len(route)):
                if(route[tmp]==0):
                    route[tmp]='↑'
                elif(route[tmp]==1):
                    route[tmp]='→'
                elif(route[tmp]==2):
                    route[tmp]='↓'
                elif(route[tmp]==3):
                    route[tmp]='←'
            print("Passage ID:{0}, passage:{1}, connectRoomID:{2}".format(Passage.id,route,Passage.connectroomid))
    print("StairsExisted:{0}".format(Room.stairexisted))

def StairsCheck(screen,nowRoomID):
    global FrameInfo
    global StairsFound
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    if(FrameInfo.get_list_of_positions_by_tile('%') != []):
        StairsFound=True
        RoomInfoList[nowRoomID].stairexisted=True


def Screenprint(screen):
    i = 0
    for i in range(24):
        print(screen[i])


def RoomInfoMake(PlayerY,PlayerX,screen,nowRoomID):
    global RoomInfoList
    global VisitedRoom
    if(VisitedRoom[nowRoomID]==False):
        #print("Room id {0} is New Room! Get LeftUp and RightBottom and doorlist!".format(nowRoomID))
        RoomInfoList[nowRoomID].leftup=getleftup(PlayerY,PlayerX,screen)
        RoomInfoList[nowRoomID].rightbottom=getrightbottom(PlayerY,PlayerX,screen)
        RoomInfoList[nowRoomID].doorlist=getdoorList(RoomInfoList[nowRoomID].leftup[0],RoomInfoList[nowRoomID].leftup[1],screen)
        VisitedRoom[nowRoomID]=True
    else:
        pass#print("Room id {0} is Not New Room. Not need make New Room Info.".format(nowRoomID))


def RightMethod():
    global FrameInfo
    global ExploreStack
    passage = []
    prevdir = -1
    CanNext = False
    CanThrough=False
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY,PlayerX = FrameInfo.get_player_pos()
    PlayerY+=1
    stackcheck=0
    stack=False
    if(FrameInfo.get_tile_below_player()!='+'):
        print("Warning! below_player tile is not door!")
            #扉からの最初の一歩
    dir = 0
    for dir in range(4):
        if(screen[PlayerY+dy[dir]][PlayerX+dx[dir]]=='#'):
            RB.send_command(command[dir])
            prevdir = dir
            passage.append(dir)
            break
    #for i in range(24):
    #    print(screen[i])

    #print("Right Method Start!")
    if(prevdir==-1):
        print("Warning! prevdir=-1!?")
        Screenprint(screen)
        for dir in range(4):
            if(screen[PlayerY+dy[dir]][PlayerX+dx[dir]]!='-' and screen[PlayerY+dy[dir]][PlayerX+dx[dir]]!='|' and \
            screen[PlayerY+dy[dir]][PlayerX+dx[dir]]!='#' and screen[PlayerY+dy[dir]][PlayerX+dx[dir]]!=' '):
                RB.send_command(command[(dir+2)%4])
                prevdir = (dir+2)%4
                passage.append((dir+2)%4)
                screen=RB.get_screen()
                Screenprint(screen)
                break
    while(True):
        if(stackcheck>=200 and stack==False):
            stack=True
            print("Warning, Maybe Stack.")
            screen = RB.get_screen()
            Screenprint(screen)
            print(passage)
            ExploreStack=True
            break
        stackcheck+=1
        CanNext = False
        CanContinue = False
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        #Screenprint(screen)
        if(FrameInfo.get_player_pos() is None):
            print("Warning! player_pos type is NoneType!")
            Screenprint(screen)
        PlayerY,PlayerX = FrameInfo.get_player_pos()
        PlayerY+=1
        #進める限り右手法で進もうとする。
        #for i in range(24):
        #    print(screen[i])
        #まずは進行方向の右側が空いているか?
        rightcommand=(prevdir+1)%4
        frontcommand=prevdir
        leftcommand=(prevdir+3)%4
        BeforeY = PlayerY
        BeforeX = PlayerX
        #sleep(0.2)
        RB.send_command(command[rightcommand])
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        AfterY,AfterX = FrameInfo.get_player_pos() 
        AfterY+=1
        if(AfterY!=BeforeY or AfterX!=BeforeX):
            prevdir = (rightcommand)%4
            passage.append(rightcommand)
            if(FrameInfo.get_tile_below_player()=='#' or FrameInfo.get_tile_below_player()==' '):
                CanNext=True
            elif(FrameInfo.get_tile_below_player()=='+'):
                CanThrough=True
        if(CanNext==False and CanThrough==False):
            RB.send_command(command[frontcommand])
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            AfterY,AfterX = FrameInfo.get_player_pos()
            AfterY+=1
            FrameInfo = RBParser.parse_screen(screen)
            #print(FrameInfo.get_tile_below_player())
            if(AfterY!=BeforeY or AfterX!=BeforeX):
                prevdir = frontcommand%4
                passage.append(frontcommand)
                if(FrameInfo.get_tile_below_player()=='#' or FrameInfo.get_tile_below_player()==' '):
                    CanNext=True
                elif(FrameInfo.get_tile_below_player()=='+'):
                    CanThrough=True
        if(CanNext==False and CanThrough==False):
            RB.send_command(command[leftcommand])
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            AfterY,AfterX = FrameInfo.get_player_pos()
            AfterY+=1
            #print(FrameInfo.get_tile_below_player())
            if(AfterY!=BeforeY or AfterX!=BeforeX):
                prevdir = (leftcommand)%4
                passage.append(leftcommand)
                if(FrameInfo.get_tile_below_player()=='#' or FrameInfo.get_tile_below_player()==' '):
                    CanNext=True
                elif(FrameInfo.get_tile_below_player()=='+'):
                    CanThrough=True

        """    
        if(screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='#' or screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='+'):
            RB.send_command(command[rightcommand])
            prevdir = (rightcommand)%4
            passage.append(rightcommand)
            if(screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='#'):
                CanNext = True
            elif(screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='+'):
                CanThrough = True

        elif(screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='#' or screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='+'):
            RB.send_command(command[frontcommand])
            prevdir = (frontcommand)%4
            passage.append(frontcommand)
            if(screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='#'):
                CanNext = True
            elif(screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='+'):
                CanThrough = True
        elif(screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='#' or screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='+'):
            RB.send_command(command[leftcommand])
            prevdir = (leftcommand)%4
            passage.append(leftcommand)
            if(screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='#'):
                CanNext = True
            elif(screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='+'):
                CanThrough = True
        """
        if(CanThrough):
            RB.send_command(command[prevdir])
            break
        #これ以上通路も扉もなく行き止まりに当たったら...
        if(CanNext==False and CanThrough==False):
            passage.reverse()
            for i in range(len(passage)):
                RB.send_command(command[(passage[i]+2)%4])#通ってきた通路を引き返し,戻っていくが...ここでも戻りつつ右手法を試みる。
                prevdir = (passage[i]+2)%4
                screen = RB.get_screen()
                rightcommand=(prevdir+1)%4
                frontcommand=prevdir
                leftcommand=(prevdir+3)%4
                #(screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='#' or screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='+') or\
                #↓引き返す時に右手法で行けるところを見つけたら、whileループの開始のところまで戻る。
                if((screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='#' or screen[PlayerY+dy[rightcommand]][PlayerX+dx[rightcommand]]=='+') or\
                (screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='#' or screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='+') or\
                (screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='#' or screen[PlayerY+dy[leftcommand]][PlayerX+dx[leftcommand]]=='+')):
                    CanContinue=True
                    del passage[:i+1]
                    passage.reverse()
                    break


            if(CanContinue):
                continue
            if(len(passage)>0):#扉から一歩以上は進めたが行き止まりだった場合
                RB.send_command(command[(passage[len(passage)-1]+2)%4])
            else:#扉に入っていきなり行き止まりだった場合
                screen = RB.get_screen()
                for i in range(4):
                    if(screen[PlayerY+dy[i]][PlayerX+dx[i]]!=' ' or screen[PlayerY+dy[i]][PlayerX+dx[i]]!='|' or\
                        screen[PlayerY+dy[i]][PlayerX+dx[i]]!='-'):
                        RB.send_command(command[i])
                        break
            passage = []
            break

    #print("Right Method End!")
    #for i in range(24):
    #    print(screen[i])
    
    return passage





def getleftup(y,x,screen):
    nowy = y
    nowx = x
    #print(screen[nowy][nowx])
    while(nowy>0 and nowx>0 and screen[nowy][nowx]!='|' and screen[nowy][nowx]!='-' and screen[nowy][nowx]!='+' and screen[nowy][nowx]!='#'):
        nowy-=1
        nowx-=1
    #print(nowy, nowx)
    #print(screen[nowy][nowx])
    if(screen[nowy][nowx]=='|'):
        while(nowy>0 and (screen[nowy][nowx]=='|' or screen[nowy][nowx]=='+') and screen[nowy][nowx+1]!='-' and screen[nowy][nowx+1]!='+'):
            nowy-=1
            #print(nowy)
    elif(screen[nowy][nowx]=='-'):
        while(nowx>0 and screen[nowy][nowx-1]!=' ' and screen[nowy][nowx-1]!='#' and screen[nowy+1][nowx]!='|' and screen[nowy+1][nowx]!='+'):
            nowx-=1
            #print(nowx)
    elif(screen[nowy][nowx]=='+'):
        if(screen[nowy][nowx-1]=='-'):
            while(nowx>0 and screen[nowy][nowx-1]!=' ' and screen[nowy][nowx-1]!='#' and screen[nowy+1][nowx]!='|' and screen[nowy+1][nowx]!='+'):
                nowx-=1
                #print(nowx)
        elif(screen[nowy-1][nowx]!=' '):
            while(nowy>0 and screen[nowy-1][nowx]!=' ' and screen[nowy-1][nowx]!='#' and screen[nowy][nowx+1]!='-' and screen[nowy][nowx+1]!='+'):
                nowy-=1
                #print(nowy)
    #print(nowy,nowx)
    return nowy,nowx

    

def getrightbottom(y,x,screen):
    nowy = y
    nowx = x
    while(nowx<80 and nowy<22 and screen[nowy][nowx]!='|' and screen[nowy][nowx]!='-' and screen[nowy][nowx]!='+' and screen[nowy][nowx]!='#'):
        nowy+=1
        nowx+=1
    if(screen[nowy][nowx]=='|'):
        while(nowy<22 and (screen[nowy][nowx]=='|' or screen[nowy][nowx]=='+') and screen[nowy][nowx-1]!='-' and screen[nowy][nowx-1]!='+'):
            nowy+=1
    elif(screen[nowy][nowx]=='-'):
        while(nowx<80 and screen[nowy][nowx+1]!=' ' and screen[nowy][nowx+1]!='#' and screen[nowy-1][nowx]!='|' and screen[nowy-1][nowx]!='+'):
            nowx+=1
    elif(screen[nowy][nowx]=='+'):
        if(screen[nowy][nowx+1]=='-'):
            while(nowx<80 and screen[nowy][nowx+1]!=' ' and screen[nowy][nowx+1]!='#' and screen[nowy-1][nowx]!='|' and screen[nowy-1][nowx]!='+'):
                nowx+=1
        elif(screen[nowy+1][nowx]!=' '):
            while(nowy<22 and screen[nowy+1][nowx]!=' ' and screen[nowy+1][nowx]!='#' and screen[nowy][nowx-1]!='-' and screen[nowy][nowx-1]!='+'):
                nowy+=1
    return nowy,nowx

#部屋のドア一覧を獲得する関数。
#左上から右→下→左→上と一周しながら見つけた順にドア情報をリストに挿入し、最後にドアリストを返す。
def getdoorList(LeftupY,LeftupX,screen):
    nowy=LeftupY
    nowx=LeftupX
    doorList=[]
    id=0
    #まずは右へ
    while(nowx<=80 and screen[nowy][nowx+1]!=' ' and screen[nowy][nowx+1]!='#'):
        nowx+=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[]))
            id+=1
        if(screen[nowy+1][nowx]=='|' or screen[nowy+1][nowx]=='+'):
            break
        #nowx+=1
    #下へ
    while(nowy<=22 and screen[nowy+1][nowx]!=' ' and screen[nowy+1][nowx]!='#'):
        nowy+=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[]))
            id+=1
        if(screen[nowy][nowx-1]=='-' or screen[nowy][nowx-1]=='+'):
            break
        #nowy+=1
    #左へ
    while(nowx>=0 and screen[nowy][nowx-1]!=' ' and screen[nowy][nowx-1]!='#'):
        nowx-=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[]))
            id+=1
        if(screen[nowy-1][nowx]=='|' or screen[nowy-1][nowx]=='+'):
            break
        #nowx-=1
    #上へ
    while(nowy>=1 and screen[nowy-1][nowx]!=' ' and screen[nowy-1][nowx]!='#'):
        nowy-=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[]))
            id+=1
        if(screen[nowy][nowx+1]=='-' or screen[nowy][nowx+1]=='+'):
            break
        #nowy-=1
    return doorList

def NotvisitedDoorCheck(nowRoomID):
    goDoorid = -1
    DoorX = -1
    DoorY = -1
    for i in range(len(RoomInfoList[nowRoomID].doorlist)):
        if(RoomInfoList[nowRoomID].doorlist[i].visited==False):
            #print("Door ID" + str(RoomInfoList[nowRoomID].doorlist[i].id) + " not Visited!")
            goDoorid = i
            DoorX = RoomInfoList[nowRoomID].doorlist[i].X
            DoorY = RoomInfoList[nowRoomID].doorlist[i].Y
            break
    return goDoorid,DoorX,DoorY

def GotoStairs(nowRoomID,StairsY,StairsX):
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    #階段のところまで地道にいき、階段にたどり着いたら階段を降りる。
    stackcheck=0
    stack=False
    StairsY+=1
    while((RB.player_pos[0]+1)!=StairsY or RB.player_pos[1]!=StairsX):
        stackcheck+=1
        PlayerY = RB.player_pos[0]+1
        PlayerX = RB.player_pos[1]
        if(PlayerX<StairsX and screen[PlayerY][PlayerX+1]!='|'):
            RB.send_command('l')
        elif(PlayerX>StairsX and screen[PlayerY][PlayerX-1]!='|'):
            RB.send_command('h')
        elif(PlayerY<StairsY and screen[PlayerY+1][PlayerX]!='-'):
            RB.send_command('j')
        elif(PlayerY>StairsY and screen[PlayerY-1][PlayerX]!='-'):
            RB.send_command('k')
        if(stackcheck>=100 and stack==False):
            stack=True
            print("Warning, Maybe Stack.")
            screen = RB.get_screen()
            Screenprint(screen)
            print("DoorList")
            print("NowRoomID: {0}".format(nowRoomID))
            print(RB.player_pos[0],RB.player_pos[1])
            print(StairsY,StairsX)
            print(screen[StairsY][StairsX])
            print(screen[RB.player_pos[0]][RB.player_pos[1]])
    RB.send_command('>')
    

def GotoDoor(nowRoomID,goDoorid,DoorY,DoorX):
    RoomInfoList[nowRoomID].doorlist[goDoorid].visited=True
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY,PlayerX = FrameInfo.get_player_pos()
    PlayerY+=1
    #まずは扉の所まで地道にいく。
    #部屋の左上右下の取得,扉リストの取得がおかしいとここでループが終わらない。(壁にガンガン)
    stackcheck=0
    stack=False
    while((RB.player_pos[0]+1)!=DoorY or RB.player_pos[1]!=DoorX):
        stackcheck+=1
        PlayerY = RB.player_pos[0]+1
        PlayerX = RB.player_pos[1]
        if(PlayerX<DoorX and screen[PlayerY][PlayerX+1]!='|'):
            RB.send_command('l')
        elif(PlayerX>DoorX and screen[PlayerY][PlayerX-1]!='|'):
            RB.send_command('h')
        elif(PlayerY<DoorY and screen[PlayerY+1][PlayerX]!='-'):
            RB.send_command('j')
        elif(PlayerY>DoorY and screen[PlayerY-1][PlayerX]!='-'):
            RB.send_command('k')
        if(stackcheck>=100 and stack==False):
            stack=True
            print("Warning, Maybe Stack.")
            screen = RB.get_screen()
            Screenprint(screen)
            print("DoorList")
            print("NowRoomID: {0}".format(nowRoomID))
            for ridx in range(len(RoomInfoList)):
                if(VisitedRoom[ridx]):
                    print("RoomID: {0}".format(ridx))
                    print("LeftUp Y {0}, X{1},".format(RoomInfoList[ridx].leftup[0],RoomInfoList[ridx].leftup[1]))
                    print("RightBottom Y {0}, X{1},".format(RoomInfoList[ridx].rightbottom[0],RoomInfoList[ridx].rightbottom[1]))
                    i=0
                    for i in range(len(RoomInfoList[ridx].doorlist)):
                        DoorID= RoomInfoList[ridx].doorlist[i].id
                        DoorX = RoomInfoList[ridx].doorlist[i].X
                        DoorY = RoomInfoList[ridx].doorlist[i].Y
                        CanThrough = len(RoomInfoList[ridx].doorlist[i].passagelist)==0 and RoomInfoList[ridx].doorlist[i].visited
                        print(DoorID,DoorY,DoorX,CanThrough)
            print("Now PlayerY,PlayerX : ({0},{1}) Aiming DoorID {2}".format(PlayerY,PlayerX,goDoorid))

#通ったことのある通路の情報に従って別の部屋に移動する。
def move(passage):
    global RoomInfoList
    global FrameInfo
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    for i in range(len(passage)):
        RB.send_command(command[passage[i]])
    RB.send_command(command[passage[-1]])

#今の部屋から行ける扉について,行ったことのない扉を探索する。
#即ち,NNが新しい場所へ探索すると決めたら、この関数を実行してもらう。
def explore(nowRoomID,GoDoorID,Beforepassagecount):
    global RoomInfoList
    global FrameInfo
    goDoorid = GoDoorID
    Passage = []
    screen = RB.get_screen()
    #print("Explore Before")
    #for i in range(24):
    #    print(screen[i])
    i=0
    FrameInfo = RBParser.parse_screen(screen)
    #print("Explore Start!")
    #for i in range(24):
    #    print(screen[i])
    """
    for i in range(len(RoomInfoList[nowRoomID].doorlist)):
        if(RoomInfoList[nowRoomID].doorlist[i].visited==False):
            #print("Door ID" + str(RoomInfoList[nowRoomID].doorlist[i].id) + " not Visited!")
            goDoorid = i
            DoorX = RoomInfoList[nowRoomID].doorlist[i].X
            DoorY = RoomInfoList[nowRoomID].doorlist[i].Y
            break
    if(goDoorid!=-1):
        RoomInfoList[nowRoomID].doorlist[goDoorid].visited=True
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        PlayerY,PlayerX = FrameInfo.get_player_pos()
        PlayerY+=1
        #まずは扉の所まで地道にいく。
        #部屋の左上右下の取得,扉リストの取得がおかしいとここでループが終わらない。(壁にガンガン)
        stackcheck=0
        stack=False
        while((RB.player_pos[0]+1)!=DoorY or RB.player_pos[1]!=DoorX):
            stackcheck+=1
            PlayerY = RB.player_pos[0]+1
            PlayerX = RB.player_pos[1]
            if(PlayerX<DoorX and screen[PlayerY][PlayerX+1]!='|'):
                RB.send_command('l')
            elif(PlayerX>DoorX and screen[PlayerY][PlayerX-1]!='|'):
                RB.send_command('h')
            elif(PlayerY<DoorY and screen[PlayerY+1][PlayerX]!='-'):
                RB.send_command('j')
            elif(PlayerY>DoorY and screen[PlayerY-1][PlayerX]!='-'):
                RB.send_command('k')
            if(stackcheck>=100 and stack==False):
                stack=True
                print("Warning, Maybe Stack.")
                screen = RB.get_screen()
                Screenprint(screen)
                print("DoorList")
                print("NowRoomID: {0}".format(nowRoomID))
                for ridx in range(len(RoomInfoList)):
                    if(VisitedRoom[ridx]):
                        print("RoomID: {0}".format(ridx))
                        print("LeftUp Y {0}, X{1},".format(RoomInfoList[ridx].leftup[0],RoomInfoList[ridx].leftup[1]))
                        print("RightBottom Y {0}, X{1},".format(RoomInfoList[ridx].rightbottom[0],RoomInfoList[ridx].rightbottom[1]))
                        i=0
                        for i in range(len(RoomInfoList[ridx].doorlist)):
                            DoorID= RoomInfoList[ridx].doorlist[i].id
                            DoorX = RoomInfoList[ridx].doorlist[i].X
                            DoorY = RoomInfoList[ridx].doorlist[i].Y
                            CanThrough = len(RoomInfoList[ridx].doorlist[i].passagelist)==0 and RoomInfoList[ridx].doorlist[i].visited
                            print(DoorID,DoorY,DoorX,CanThrough)
                print("Now PlayerY,PlayerX : ({0},{1}) Aiming DoorID {2}".format(PlayerY,PlayerX,goDoorid))
    """




    #print("Get Door!")
    screen = RB.get_screen()

    #扉までついたら,右手法に従って通路を進んでもらう.
    #通路を実際に進むところは丸ごとRightMethod内で行なっている
    Passage = RightMethod()
    if(ExploreStack==True):
        return [-1]
    #右手法が終わった後,辿ってきた経路について...
    #①空の場合,そもそも別の部屋に行けないので,この部屋のこの扉そのものをなかった事にする。
    if(Passage==[]):
        RoomInfoList[nowRoomID].doorlist[goDoorid].visited=True
    #②空では無い場合、まず辿り着いた部屋は行った事があるかどうかを確認する関数checkVisitedを使ってたどり着いた部屋のIDを取得する。
    #部屋のIDは特に決まった順序があるわけではなく,見つけた順に若いIDが割り当てられる.
    else:
        PlayerY = RB.player_pos[0]+1
        PlayerX = RB.player_pos[1]
        goRoomid = checkVisited(PlayerY,PlayerX)
        #print(goRoomid)
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        Afterpassagecount = FrameInfo.get_tile_count('#')
        RoomInfoMake(PlayerY,PlayerX,screen,goRoomid)
        #print(Afterpassagecount-Beforepassagecount,len(Passage)-1)
        didx = 0
        if(Afterpassagecount-Beforepassagecount==len(Passage)-1):
            #print("One Road!")
            for door in RoomInfoList[goRoomid].doorlist:
                #door = RoomInfoList[goRoomid].doorlist[d]
                if(door.Y == PlayerY+dy[(Passage[-1]+2)%4] and door.X == PlayerX+dx[(Passage[-1]+2)%4]):
                    reversePassage = Passage.copy()
                    reversePassage.reverse()
                    for r in range(len(reversePassage)):
                        reversePassage[r]= (reversePassage[r]+2)%4
                    door.visited=True
                    door.passagelist.append(PassageInfo(len(RoomInfoList[goRoomid].doorlist[didx].passagelist),reversePassage,nowRoomID))
                didx+=1
                #print(door.id, door.Y, door.X, door.visited)
            #VisitedRoom[goRoomid]=True
            #print("After Explore")
            #for k in range(24):
            #    print(screen[k])
        RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist.append(PassageInfo(len(RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist),Passage,goRoomid))
    #print("Explore End!")
    screen = RB.get_screen()
    #for i in range(24):
    #    print(screen[i])
    return Passage

#探索後に辿り着いた部屋のIDを調べる。
#行ったことのある部屋ならその部屋のIDを返し,行った事が無い部屋なら,現在の部屋ID+1を設定する。
def checkVisited(playerY,playerX):
    global maxRoomID
    for i in range(15):
        if((playerY>=RoomInfoList[i].leftup[0] and playerY<=RoomInfoList[i].rightbottom[0]) and \
            (playerX>=RoomInfoList[i].leftup[1] and playerX<=RoomInfoList[i].rightbottom[1])):
            return RoomInfoList[i].id
    maxRoomID+=1
    return maxRoomID

def eval_network(net, net_input):
    assert (len(net_input) == 363)
    #assert (result[0] >= -1.0 or result[0] <= 1.0)
    return np.argmax(net.activate(net_input))


def eval_single_genome(genome, genome_config):
    global RB
    global RBParser
    global FrameInfo
    global maxRoomID
    global RoomInfoList
    global VisitedRoom
    global StairsFound
    global ExploreStack
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0
    Roguetime=0
    for _ in range(run_neat_base.n):
        #print("--> Starting new episode")
        ExploreStack = False
        MovingStack = 0
        StairsFound = False
        maxRoomID = -1
        t=0
        Roguetime+=1
        #print(Roguetime)
        random.seed()
        rn=random.random()*1000000
        rn=math.floor(rn)
        run_neat_base.env.seed(rn)
        run_neat_base.env.reset()
        observation = run_neat_base.env.reset()
        inputs =observation
        #inputs = inputs.flatten()
        inputs = inputs.reshape(inputs.size,1)
        nowRoomID=0
        #action = eval_network(net, inputs)
        RB = run_neat_base.env.unwrapped.rb
        RBParser = RB.parser
        RBParser.reset()
        RoomInfoList.clear()
        Playery = RB.player_pos[0]+1
        Playerx = RB.player_pos[1]
        VisitedRoom = np.array([False]*15)
        if(t==0):
            for k in range(15):
                RoomInfoList.append(RoomInfo(k,(-1,-1),(-1,-1),[],[],False))
        done = False
        while not done:
            MovingStack = 0
            #run_neat_base.env.render()
            t+=1
            #observation, reward, done, info = run_neat_base.env.step(action)

            
            #inputs = observation
            #inputs = inputs.reshape(inputs.size,1)
            #RogueBoxオブジェクトを呼び出すときは,env.unwrapped.rb
            screen = RB.get_screen()
            Playery = RB.player_pos[0]+1
            Playerx = RB.player_pos[1]
            #sleep(1)
            #for i in range(24):
            #    print(screen[i])
            #print("PlayerY: " + str(Playery) + " PlayerX: " + str(Playerx))
            nowRoomID = checkVisited(Playery,Playerx)
            #print(nowRoomID)
            FrameInfo = RBParser.parse_screen(screen)
            RoomInfoMake(Playery,Playerx,screen,nowRoomID)
            if(StairsFound==False):
                StairsCheck(screen,nowRoomID)
            if(RoomInfoList[nowRoomID].stairexisted==True):
                FrameInfo = RBParser.parse_screen(screen)
                StairsCoord=FrameInfo.get_list_of_positions_by_tile('%')
                (StairsY,StairsX)=StairsCoord[0]
                GotoStairs(nowRoomID,StairsY,StairsX)
                break
            #print("Room "+ str(nowRoomID) + " leftup: " + str(RoomInfoList[nowRoomID].leftup))
            #print("Room "+ str(nowRoomID) + " rightbottom: " + str(RoomInfoList[nowRoomID].rightbottom))
            #for i in range(len(RoomInfoList[nowRoomID].doorlist)):
            #    print(RoomInfoList[nowRoomID].doorlist[i].id,RoomInfoList[nowRoomID].doorlist[i].Y,\
            #        RoomInfoList[nowRoomID].doorlist[i].X,RoomInfoList[nowRoomID].doorlist[i].visited)

            #次の部屋へいくための道を取得してもらう。
            #尚、ここで求めた道が明確に次の部屋と一本道である場合、次の部屋の対応する扉について、
            #求めた道を反転した道を次の部屋の既知の扉と道の情報とする。
            #明示的に一本道であると確定するには,get_tile_count('#')で探索前後における#の数を数える。
            GoDoorID,GoDoorX,GoDoorY = NotvisitedDoorCheck(nowRoomID)
            Beforepassagecount = FrameInfo.get_tile_count('#')
            
            """
            print("----------Before Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """

            if(GoDoorID!=-1 and GoDoorX!=-1 and GoDoorY!=-1):
                #print("There is unvisited Door! Explore Start!")
                GotoDoor(nowRoomID,GoDoorID,GoDoorY,GoDoorX)
                Passage = explore(nowRoomID,GoDoorID,Beforepassagecount)
                if(len(Passage)>0 and Passage[0]==-1):
                    print("Can't explore not visited door. End.")
                    break
            else:
                #print("Unvisited Door is nothing. Random move Start!")
                GoDoorID=-1
                while(True):
                    MovingStack+=1
                    if(MovingStack>=100):
                        print("Warning, Maybe Stack.")
                        screen = RB.get_screen()
                        Screenprint(screen)
                        Playery=RB.player_pos[0]
                        Playerx=RB.player_pos[1]
                        nowRoomID=checkVisited(Playery,Playerx)
                        RoominfoPrint(nowRoomID)
                        break
                    if(len(RoomInfoList[nowRoomID].doorlist)==0):
                        print("Warning len(Doorlist)=0.")
                        screen = RB.get_screen()
                        Screenprint(screen)
                        Playery=RB.player_pos[0]
                        Playerx=RB.player_pos[1]
                        nowRoomID=checkVisited(Playery,Playerx)
                        RoominfoPrint(nowRoomID)
                        print(rn)
                        break
                    GoDoorID = random.randrange(len(RoomInfoList[nowRoomID].doorlist))
                    if(len(RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist)>0):
                        break
                if(MovingStack>=100 or len(RoomInfoList[nowRoomID].doorlist)==0):
                    break
                GoDoorX = RoomInfoList[nowRoomID].doorlist[GoDoorID].X
                GoDoorY = RoomInfoList[nowRoomID].doorlist[GoDoorID].Y
                Gopassage = RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist[0].passage
                GotoDoor(nowRoomID,GoDoorID,GoDoorY,GoDoorX)
                move(Gopassage)

            #if(len(Passage)!=0):
            #    print("Get Next Room!")
            #else:
            #    print("Can't get Next Room...")
            Playery = RB.player_pos[0]+1
            Playerx = RB.player_pos[1]
            screen = RB.get_screen()
            nowRoomID = checkVisited(Playery,Playerx)

            """
            print("----------After Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """
            
            #sleep(2)
            
            #sleep(5)
            #action = eval_network(net, inputs)
            
            #total_reward += reward

            if done or t>=30:
                #print("<-- Episode finished after {} time-steps with reward {}".format(t + 1, total_reward))
                break
    #print(total_reward / run_neat_base.n)
    #print("<-- Episode finished after average {} time-steps with reward {}".format(t + 1//5, total_reward / run_neat_base.n))
    #print("Fin")
    return total_reward / run_neat_base.n


def learn(env,config_path):
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment=env,
                      config_path=config_path)

