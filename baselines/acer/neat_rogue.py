import sys
sys.path.append('.../')
import MOneat as neat
import numpy as np
import types
from . import run_neat_base
import random
import math
from time import sleep
import copy
import re
from rogueinabox_lib.parser import RogueParser
RB = None
RBParser = None
FrameInfo = None
SPACE=chr(20)
ESC=chr(27)
STAR=chr(30)
dx = [ 0, 1, 0, -1]
dy = [ -1, 0, 1, 0]
command =['k','l','j','h']
atoz = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
Foodidx=-1
FoodNum=0
ExtraExplore=0
Arrowidx=[]
RoomInfoList = []
maxRoomID = -1
RightActnum = 0
ExploreActnum= 0
VisitedRoom = None
StairsFound = False
ExploreStack = False
GlobalStackCheck=False
class RoomInfo:
    def __init__(self,id,leftup,rightbottom,doorlist,knowndoorlist,stairexisted,stairsRoomdis):
        self.id = id
        self.leftup = leftup
        self.rightbottom = rightbottom
        self.doorlist = doorlist
        self.knowndoorlist = knowndoorlist
        self.stairexisted = stairexisted
        self.stairsRoomdis = stairsRoomdis
        self.itemList = []

class DoorInfo:
    def __init__(self,id,Y,X,visited,passagelist,useless):
        self.id = id
        self.Y = Y
        self.X = X
        self.visited = False
        self.passagelist = passagelist
        self.useless = useless

class PassageInfo:
    def __init__(self,id,passage,connectroomid):
        self.id = id
        self.passage = passage
        self.connectroomid = connectroomid

def StairsRoomdisDFS(nowRoomID, visitedDFSArr, Prevdis):
    RoomInfoList[nowRoomID].stairsRoomdis=Prevdis
    nxtDFSvisitedArr=visitedDFSArr.copy()
    nxtDFSvisitedArr[nowRoomID]=True
    for d in range(len(RoomInfoList[nowRoomID].doorlist)):
        door=RoomInfoList[nowRoomID].doorlist[d]
        if(door.visited):
            nxtRoomID=door.passagelist[0].connectroomid
            if(not nxtDFSvisitedArr[nxtRoomID] and RoomInfoList[nxtRoomID].stairsRoomdis>Prevdis+1):
                StairsRoomdisDFS(nxtRoomID, nxtDFSvisitedArr, Prevdis+1)
        else:
            continue

def RoomObjectSearch(screen, leftup,rightbottom,obj):
    ret=0
    if(obj!="monster"):
        for y in range(leftup[0],rightbottom[0]):
            for x in range(leftup[1],rightbottom[1]):
                if(screen[y][x]==obj):
                    ret+=1
    else:
        for y in range(leftup[0],rightbottom[0]):
            for x in range(leftup[1],rightbottom[1]):
                if(screen[y][x].isupper()):
                    ret+=1
    return ret

def PickupSearch(nowRoomID):
    room = RoomInfoList[nowRoomID]
    leftup = room.leftup
    rightbottom = rool.rightbottom
    screen = RB.get_screen()
    room.itemList=[]
    for y in range(leftup[0],rightbottom[0]):
        for x in range(leftup[1], rightbottom[1]):
            if(screen[y][x]=='!' or screen[y][x]=='?' or screen[y][x]=='/' \
                or screen[y][x]==')' or screen[y][x]=='*' or screen[y][x]==':'):
                room.itemList.append((y,x,screen[y][x]))

def CheckFoodnum():
    global Foodidx
    RB.send_command('e')
    #print("NowFoodNum: {0}".format(RB.FoodNum))
    return RB.FoodNum
    """
    sleep(0.01)
    RB._update_screen()
    screen=RB.get_screen()
    Screenprint(screen)
    sleep(0.04)
    RB.pipe.write(STAR.encode())
    RB._update_screen()
    screen=RB.get_screen()
    Screenprint(screen)
    for y in range(len(screen)):
        if('food' in screen[y]):
            Foodidx=atoz[y]
            if('Some' in screen[y]):
                RB.pipe.write(' '.encode())
                RB.pipe.write(ESC.encode())
                return 1
            RB.pipe.write(' '.encode())
            RB.pipe.write(ESC.encode())
            return int(re.sub("\\D","",screen[y]))
    RB.pipe.write(' '.encode())
    RB.pipe.write(ESC.encode())
    return 0
    """
        
def EatFood():
    RB.send_command('e')
    RB.send_command(Foodidx)




def MakeInput(nowRoomID):
    global FrameInfo
    global FoodNum
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    input = np.full(27,-1.0,dtype=float)
    if(StairsFound):
        input[0]=1.0
    Room = RoomInfoList[nowRoomID]
    i=0
    for d in range(len(Room.doorlist)):
        if(Room.doorlist[d].visited):
            input[i*5+1]=1.0
            if(Room.doorlist[d].useless==False):
                input[i*5+4]=1.0       
                nxtroom = RoomInfoList[Room.doorlist[d].passagelist[0].connectroomid]
                for nd in range(len(nxtroom.doorlist)):
                    if(not nxtroom.doorlist[nd].visited):
                        input[i*5+2]+=1.0
                input[i*5+3]=nxtroom.stairsRoomdis       
        else:
            input[i*5+1]=-1.0
        i+=1
    input[20]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,':')
    input[21]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,'!')
    input[22]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,'?')
    input[23]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,')')
    input[24]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,'monster')
    input[25]=RoomObjectSearch(screen,Room.leftup,Room.rightbottom,'*')
    input[26]=CheckFoodnum()
    FoodNum=input[26]
    return input
                

def RoominfoPrint(nowRoomID):
    print("RoomInformation print!")
    print("Now Room ID: {0}".format(nowRoomID))
    print("LeftUp Y {0}, X{1},".format(RoomInfoList[nowRoomID].leftup[0],RoomInfoList[nowRoomID].leftup[1]))
    print("RightBottom Y {0}, X{1},".format(RoomInfoList[nowRoomID].rightbottom[0],RoomInfoList[nowRoomID].rightbottom[1]))
    Room = RoomInfoList[nowRoomID]
    for d in range(len(Room.doorlist)):
        door = Room.doorlist[d]
        print("Door ID:{0}, Coordinates:({1},{2}), Visited:{3}, Useless:{4}.".format(door.id,door.Y,door.X,door.visited,door.useless))
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
    if(not StairsFound and FrameInfo.get_list_of_positions_by_tile('%') != []):
        StairsFound=True
        RoomInfoList[nowRoomID].stairexisted=True
        RoomInfoList[nowRoomID].stairsRoomdis=0
        DFSArr=np.array([False]*15)
        StairsRoomdisDFS(nowRoomID, DFSArr, 0)


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

def StairsRoomdisCheck(nowRoomID,Prevdis):
    if(Prevdis==100):
        return
    RoomInfoList[nowRoomID].stairsRoomdis=min(RoomInfoList[nowRoomID].stairsRoomdis,Prevdis+1)


def RightMethod(playery,playerx):
    global FrameInfo
    global ExploreStack
    global GlobalStackCheck
    passage = []
    prevdir = -1
    CanNext = False
    CanThrough=False
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    if(FrameInfo.get_player_pos() ==  None):
        screen_refresh()
    PlayerY,PlayerX = playery, playerx
    #PlayerY+=1
    stackcheck=0
    stack=False
    if(FrameInfo.get_tile_below_player()!='+'):
        print("Warning! below_player tile is not door!")
            #扉からの最初の一歩
    dir = 0
    for dir in range(4):
        if(screen[PlayerY+dy[dir]][PlayerX+dx[dir]]=='#' or screen[PlayerY+dy[dir]][PlayerX+dx[dir]]==' '):
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
        if(RB.game_over()==True):
            break
        if(stackcheck>=200 and stack==False):
            stack=True
            print("Warning, Maybe Stack.")
            screen = RB.get_screen()
            Screenprint(screen)
            print(passage)
            ExploreStack=True
            GlobalStackCheck=True
            break
        stackcheck+=1
        CanNext = False
        CanContinue = False
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        #Screenprint(screen)
        if(FrameInfo.get_player_pos() ==  None):
            print("Warning! player_pos type is NoneType!")
            print(len(screen),len(screen[0]))
            Screenprint(screen)
            screen_refresh()
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
        if(FrameInfo.get_player_pos() ==  None):
            print("Warning! player_pos type is NoneType!")
            print(len(screen),len(screen[0]))
            Screenprint(screen)
            screen_refresh()
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
            if(FrameInfo.get_player_pos() ==  None):
                print("Warning! player_pos type is NoneType!")
                print(len(screen),len(screen[0]))
                Screenprint(screen)
                screen_refresh()
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
            doorList.append(DoorInfo(id,nowy,nowx,False,[],False))
            id+=1
        if(screen[nowy+1][nowx]=='|' or screen[nowy+1][nowx]=='+'):
            break
        #nowx+=1
    #下へ
    while(nowy<=22 and screen[nowy+1][nowx]!=' ' and screen[nowy+1][nowx]!='#'):
        nowy+=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[],False))
            id+=1
        if(screen[nowy][nowx-1]=='-' or screen[nowy][nowx-1]=='+'):
            break
        #nowy+=1
    #左へ
    while(nowx>=0 and screen[nowy][nowx-1]!=' ' and screen[nowy][nowx-1]!='#'):
        nowx-=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[],False))
            id+=1
        if(screen[nowy-1][nowx]=='|' or screen[nowy-1][nowx]=='+'):
            break
        #nowx-=1
    #上へ
    while(nowy>=1 and screen[nowy-1][nowx]!=' ' and screen[nowy-1][nowx]!='#'):
        nowy-=1
        if(screen[nowy][nowx]=='+'):
            doorList.append(DoorInfo(id,nowy,nowx,False,[],False))
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

def GotoObject(nowRoomID,aimY,aimX,object):
    global GlobalStackCheck
    screen = RB.get_screen()
    if(RoomInfoList[nowRoomID].stairexisted==False):
        return False
    #階段のところまで地道にいき、階段にたどり着いたら階段を降りる。
    stackcheck=0
    stack=False
    aimY+=1
    PlayerY = -1
    PlayerX = -1
    while(True):
        if(len(RB.player_pos)>0):
            PlayerY = RB.player_pos[0]+1
            PlayerX = RB.player_pos[1]
        else:
            screen_refresh()
            PlayerY = RB.player_pos[0]+1
            PlayerX = RB.player_pos[1]
        if(PlayerY != aimY or PlayerX != aimX):
            stackcheck+=1
            if(PlayerX<aimX and PlayerY<aimY and \
                screen[PlayerY+1][PlayerX+1]!='|' and screen[PlayerY+1][PlayerX+1]!='-'):
                RB.send_command('m')
                RB.send_command('n')
                PlayerX+=1
                PlayerY+=1
            elif(PlayerX>aimX and PlayerY<aimY and \
                screen[PlayerY+1][PlayerX-1]!='|' and screen[PlayerY+1][PlayerX-1]!='-'):
                RB.send_command('m')
                RB.send_command('b')
                PlayerX-=1
                PlayerY+=1
            elif(PlayerX<aimX and PlayerY>aimY and \
                screen[PlayerY-1][PlayerX+1]!='|' and screen[PlayerY-1][PlayerX+1]!='-'):
                RB.send_command('m')
                RB.send_command('u')
                PlayerX+=1
                PlayerY-=1
            elif(PlayerX<aimX and PlayerY<aimY and \
                screen[PlayerY-1][PlayerX-1]!='|' and screen[PlayerY-1][PlayerX-1]!='-'):
                RB.send_command('m')
                RB.send_command('y')
                PlayerX-=1
                PlayerY-=1
            elif(PlayerX<aimX and screen[PlayerY][PlayerX+1]!='|'):
                RB.send_command('m')
                RB.send_command('l')
                PlayerX+=1
            elif(PlayerX>aimX and screen[PlayerY][PlayerX-1]!='|'):
                RB.send_command('m')
                RB.send_command('h')
                PlayerX-=1
            elif(PlayerY<aimY and screen[PlayerY+1][PlayerX]!='-'):
                RB.send_command('m')
                RB.send_command('j')
                PlayerY+=1
            elif(PlayerY>aimY and screen[PlayerY-1][PlayerX]!='-'):
                RB.send_command('m')
                RB.send_command('k')
                PlayerY-=1
            if(stackcheck>=100 and stack==False):
                stack=True
                print("Warning, Maybe Stack while go to stairs.")
                screen = RB.get_screen()
                Screenprint(screen)
                print("DoorList")
                print("NowRoomID: {0}".format(nowRoomID))
                print("Now Player:({0},{1})".format(RB.player_pos[0],RB.player_pos[1]))
                print("Stairs:({0},{1})".format(aimY,aimX))
                for ridx in range(len(RoomInfoList)):
                    if(VisitedRoom[ridx]):
                        print("RoomID: {0}".format(ridx))
                        print("LeftUp Y {0}, X{1},".format(RoomInfoList[ridx].leftup[0],RoomInfoList[ridx].leftup[1]))
                        print("RightBottom Y {0}, X{1},".format(RoomInfoList[ridx].rightbottom[0],RoomInfoList[ridx].rightbottom[1]))
                        print("Stairs Existed{0}".format(RoomInfoList[ridx].stairexisted))
                        i=0
                        for i in range(len(RoomInfoList[ridx].doorlist)):
                            DoorID= RoomInfoList[ridx].doorlist[i].id
                            DoorX = RoomInfoList[ridx].doorlist[i].X
                            DoorY = RoomInfoList[ridx].doorlist[i].Y
                            CanThrough = len(RoomInfoList[ridx].doorlist[i].passagelist)==0 and RoomInfoList[ridx].doorlist[i].visited
                            print(DoorID,DoorY,DoorX,CanThrough)
                GlobalStackCheck=True
                break
        else:
            break
    if(len(RB.player_pos)<1):
        screen=RB.get_screen()
        Screenprint(screen)
        RB.send_command('i')
        sleep(RB.busy_wait_seconds*400)
        screen=RB.get_screen()
        Screenprint(screen)
        return False
    if(object=='%'):
        RB.send_command('>')
    else:
        RB.send_command(',')
    return True
    

def GotoDoor(nowRoomID,goDoorid,DoorY,DoorX):
    global GlobalStackCheck
    RoomInfoList[nowRoomID].doorlist[goDoorid].visited=True
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY = -1
    PlayerX = -1
    #まずは扉の所まで地道にいく。
    #部屋の左上右下の取得,扉リストの取得がおかしいとここでループが終わらない。(壁にガンガン)
    stackcheck=0
    stack=False
    while(True):
        if(len(RB.player_pos)>0):
            PlayerY = RB.player_pos[0]+1
            PlayerX = RB.player_pos[1]
        else:
            screen_refresh()
            PlayerY = RB.player_pos[0]+1
            PlayerX = RB.player_pos[1]
        print("Player(Y,X):({0},{1})".format(PlayerY,PlayerX))
        if(PlayerY != DoorY or PlayerX != DoorX):
            if(RB.game_over()==True):
                break
            stackcheck+=1
            if(PlayerX<DoorX and PlayerY<DoorY and \
                screen[PlayerY+1][PlayerX+1]!='|' and screen[PlayerY+1][PlayerX+1]!='-'):
                RB.send_command('m')
                RB.send_command('n')
                PlayerX+=1
                PlayerY+=1
            elif(PlayerX>DoorX and PlayerY<DoorY and \
                screen[PlayerY+1][PlayerX-1]!='|' and screen[PlayerY+1][PlayerX-1]!='-'):
                RB.send_command('m')
                RB.send_command('b')
                PlayerX-=1
                PlayerY+=1
            elif(PlayerX<DoorX and PlayerY>DoorY and \
                screen[PlayerY-1][PlayerX+1]!='|' and screen[PlayerY-1][PlayerX+1]!='-'):
                RB.send_command('m')
                RB.send_command('u')
                PlayerX+=1
                PlayerY-=1
            elif(PlayerX<DoorX and PlayerY<DoorY and \
                screen[PlayerY-1][PlayerX-1]!='|' and screen[PlayerY-1][PlayerX-1]!='-'):
                RB.send_command('m')
                RB.send_command('y')
                PlayerX-=1
                PlayerY-=1
            elif(PlayerX<DoorX and screen[PlayerY][PlayerX+1]!='|'):
                RB.send_command('m')
                RB.send_command('l')
                PlayerX+=1
            elif(PlayerX>DoorX and screen[PlayerY][PlayerX-1]!='|'):
                RB.send_command('m')
                RB.send_command('h')
                PlayerX-=1
            elif(PlayerY<DoorY and screen[PlayerY+1][PlayerX]!='-'):
                RB.send_command('m')
                RB.send_command('j')
                PlayerY+=1
            elif(PlayerY>DoorY and screen[PlayerY-1][PlayerX]!='-'):
                RB.send_command('m')
                RB.send_command('k')
                PlayerY-=1
            if(RB.game_over()==True):
                break
            if(stackcheck>=100 and stack==False):
                stack=True
                GlobalStackCheck=True
                print("Warning, Maybe Stack while going to door.")
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
                break
        else:
            break
    if(len(RB.player_pos)<1):
        screen_refresh()

    return PlayerY,PlayerX

def screen_refresh():
    screen=RB.get_screen()
    Screenprint(screen)
    RB.send_command('i')
    sleep(RB.busy_wait_seconds*400)
    screen=RB.get_screen()
    Screenprint(screen)

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
def explore(nowRoomID,GoDoorID,Beforepassagecount,playery,playerx):
    global RoomInfoList
    global FrameInfo
    global GlobalStackCheck
    goDoorid = GoDoorID
    Passage = []
    screen = RB.get_screen()
    i=0
    FrameInfo = RBParser.parse_screen(screen)
    screen = RB.get_screen()

    #扉までついたら,右手法に従って通路を進んでもらう.
    #通路を実際に進むところは丸ごとRightMethod内で行なっている
    Passage = RightMethod(playery,playerx)
    if(ExploreStack==True):
        GlobalStackCheck=True
        return [-1]
    #右手法が終わった後,辿ってきた経路について...
    #①空の場合,そもそも別の部屋に行けないので,この部屋のこの扉そのものをなかった事にする。
    if(Passage==[]):
        RoomInfoList[nowRoomID].doorlist[goDoorid].visited=True
        RoomInfoList[nowRoomID].doorlist[goDoorid].useless=True
    #②空では無い場合、まず辿り着いた部屋は行った事があるかどうかを確認する関数checkVisitedを使ってたどり着いた部屋のIDを取得する。
    #部屋のIDは特に決まった順序があるわけではなく,見つけた順に若いIDが割り当てられる.
    else:
        PlayerY = RB.player_pos[0]+1
        PlayerX = RB.player_pos[1]
        goRoomid = checkVisited(PlayerY,PlayerX)
        if(goRoomid==nowRoomID):
            RoomInfoList[nowRoomID].doorlist[goDoorid].useless=True
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        Afterpassagecount = FrameInfo.get_tile_count('#')
        RoomInfoMake(PlayerY,PlayerX,screen,goRoomid)
        didx = 0
        if(Afterpassagecount-Beforepassagecount==len(Passage)-1):
            #print("One Road!")
            for door in RoomInfoList[goRoomid].doorlist:
                if(door.Y == PlayerY+dy[(Passage[-1]+2)%4] and door.X == PlayerX+dx[(Passage[-1]+2)%4]):
                    reversePassage = Passage.copy()
                    reversePassage.reverse()
                    for r in range(len(reversePassage)):
                        reversePassage[r]= (reversePassage[r]+2)%4
                    door.visited=True
                    door.passagelist.append(PassageInfo(len(RoomInfoList[goRoomid].doorlist[didx].passagelist),reversePassage,nowRoomID))
                didx+=1
        RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist.append(PassageInfo(len(RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist),Passage,goRoomid))
    screen = RB.get_screen()
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
    assert (len(net_input) == 27)
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
    for _ in range(run_neat_base.n):
        return RogueTrying(net)
    #print(total_reward / run_neat_base.n)
    #print("<-- Episode finished after average {} time-steps with reward {}".format(t + 1//5, total_reward / run_neat_base.n))
    #print("Fin")

def RogueTrying(net):
    fitness=[0.0,0.0,0.0]
    global RightActnum
    global ExploreActnum
    for _ in range(run_neat_base.n):
        Roguetime=0
        Initialize()
        #print("--> Starting new episode")
        MovingStack = 0
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
        Playery = RB.player_pos[0]+1
        Playerx = RB.player_pos[1]
        done = False
        while not done:
            MovingStack = 0
            #run_neat_base.env.render()
            t+=1
            #RogueBoxオブジェクトを呼び出すときは,env.unwrapped.rb
            screen = RB.get_screen()
            Playery = RB.player_pos[0]+1
            Playerx = RB.player_pos[1]
            nowRoomID = checkVisited(Playery,Playerx)
            FrameInfo = RBParser.parse_screen(screen)
            RoomInfoMake(Playery,Playerx,screen,nowRoomID)
            if(StairsFound==False):
                StairsCheck(screen,nowRoomID)
            Prevdis = RoomInfoList[nowRoomID].stairsRoomdis
            Input = MakeInput(nowRoomID)
            action = eval_network(net, Input)
            """
            print("----------Before Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """
            if(RB.game_over()==True):
                print("Game Over.")
                fitness = [ x + y for (x, y) in zip(fitness,[-50,-50,0]) ]
                break
            Safe,Stairdown = ActMethod(action,nowRoomID)
            if(GlobalStackCheck==True):
                print("Unfortunately Stacking. Retry.")
                return RogueTrying(net)
            #次の部屋へいくための道を取得してもらう。
            #尚、ここで求めた道が明確に次の部屋と一本道である場合、次の部屋の対応する扉について、
            #求めた道を反転した道を次の部屋の既知の扉と道の情報とする。
            #明示的に一本道であると確定するには,get_tile_count('#')で探索前後における#の数を数える。
            if(RB.game_over()==True):
                print("Game Over.")
                fitness = [ x + y for (x, y) in zip(fitness,[-50,-50,0]) ]
                break

            Playery = RB.player_pos[0]+1
            Playerx = RB.player_pos[1]
            screen = RB.get_screen()
            if(Safe==False):
                #print("Bad Action!")
                #fitness = [ x + y for (x, y) in zip(fitness,[-50,-50,0]) ]
                done=True
            else:
                RightActnum+=1
                if(action==4):
                    ExploreActnum+=1
            if(Safe==True and Stairdown==True):
                fitness = [ x + y for (x, y) in zip(fitness,[ExploreActnum,ExtraExplore,RightActnum]) ]
                print("Get down stairs! Add Fitness: {0}".format([ExploreActnum,ExtraExplore,RightActnum]))
                Initialize()
                t=0

            if('Hungry' in screen and FoodNum>0):
                EatFood()
            nowRoomID = checkVisited(Playery,Playerx)
            if(StairsFound==True and RoomInfoList[nowRoomID].stairsRoomdis==100):
                StairsRoomdisCheck(nowRoomID,Prevdis)

            """
            print("----------After Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """
            #sleep(0)
            if done or t>=20:
                if(Safe==True and t<20):
                    fitness = [ x + y for (x, y) in zip(fitness,[ExploreActnum,ExtraExplore,RightActnum]) ]
                    print("???")

                    #fitness = [30-ExtraExplore,ExtraExplore,RightActnum]
                if(t==20 or Safe==False):
                    fitness = [ x + y for (x, y) in zip(fitness,[-30+ExploreActnum,-50,RightActnum]) ]
                    #fitness = [-50,-50,RightActnum]
                #print(fitness)
                break
    #print(total_reward / run_neat_base.n)
    #print("<-- Episode finished after average {} time-steps with reward {}".format(t + 1//5, total_reward / run_neat_base.n))
    #print("Fin")
    return fitness

def Initialize():
    global RB
    global RBParser
    global FrameInfo
    global maxRoomID
    global RoomInfoList
    global VisitedRoom
    global StairsFound
    global ExploreStack
    global ExtraExplore
    global GlobalStackCheck
    global RightActnum
    global ExploreActnum
    ExploreStack = False
    GlobalStackCheck=False
    MovingStack = 0
    ExtraExplore=0
    RightActnum = 0
    ExploreActnum = 0
    StairsFound = False
    maxRoomID = -1
    RB = run_neat_base.env.unwrapped.rb
    RBParser = RB.parser
    RBParser.reset()
    RoomInfoList.clear()
    VisitedRoom = np.array([False]*15)
    for k in range(15):
        RoomInfoList.append(RoomInfo(k,(-1,-1),(-1,-1),[],[],False,100))

def learn(env,config_path):
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment=env,
                      config_path=config_path)

def GotoStairsAct(nowRoomID,screen):
    if(RoomInfoList[nowRoomID].stairexisted==True):
        FrameInfo = RBParser.parse_screen(screen)
        StairsCoord=FrameInfo.get_list_of_positions_by_tile('%')
        (StairsY,StairsX)=StairsCoord[0]
        GotoObject(nowRoomID,StairsY,StairsX,'%')
        return True
    else:
        return False

def GotoKnownRoomAct(nowRoomID,GoDoorID):
    #print(len(RoomInfoList[nowRoomID].doorlist))
    if(GoDoorID+1>len(RoomInfoList[nowRoomID].doorlist)):
        #print("GoDoorID is too large!")
        return False
    elif(not RoomInfoList[nowRoomID].doorlist[GoDoorID].visited):
        #print("Door ID:GoDoorID didn't visit!")
        return False
    GoDoorX = RoomInfoList[nowRoomID].doorlist[GoDoorID].X
    GoDoorY = RoomInfoList[nowRoomID].doorlist[GoDoorID].Y
    Gopassage = RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist[0].passage
    GotoDoor(nowRoomID,GoDoorID,GoDoorY,GoDoorX)
    if(RB.game_over()==True):
        return False
    move(Gopassage)
    return True

def GotoStairsRoomAct(nowRoomID):
    if(StairsFound==False or RoomInfoList[nowRoomID].stairexisted==True):
        return False
    Mindis=50
    GoDoorID=-1
    for d in range(len(RoomInfoList[nowRoomID].doorlist)):
        if(RoomInfoList[nowRoomID].doorlist[GoDoorID].visited and \
            len(RoomInfoList[nowRoomID].doorlist[d].passagelist)>0 and \
            RoomInfoList[RoomInfoList[nowRoomID].doorlist[d].passagelist[0].connectroomid].stairsRoomdis<Mindis):
            Mindis=RoomInfoList[RoomInfoList[nowRoomID].doorlist[d].passagelist[0].connectroomid].stairsRoomdis
            GoDoorID=d
    if(GoDoorID==-1):
        return False
    return GotoKnownRoomAct(nowRoomID,GoDoorID)
    

def FightAct(screen,nowRoomID):
    Enemynum=RoomObjectSearch(screen,RoomInfoList[nowRoomID].leftup,RoomInfoList[nowRoomID].rightbottom,'monster')
    if(Enemynum==0):
        return False
    for _ in range(30):
        Playery = RB.player_pos[0]+1
        Playerx = RB.player_pos[1]
        screen = RB.get_screen()
        for k in range(4):
            if(screen[Playery+dy[k]][Playerx+dx[k]].isupper()):
                CloseFight(k)
                screen = RB.get_screen()
                Screenprint(screen)        
        Enemynum=RoomObjectSearch(screen,RoomInfoList[nowRoomID].leftp,RoomInfoList[nowRoomID].rightbottom,'monster')
        if(Enemynum==0):
            break
    return True




def CloseFight(Direction):
    RB.send_command('f')
    RB.send_command(command[Direction])


def ExploreAct(nowRoomID):
    GoDoorID,GoDoorX,GoDoorY = NotvisitedDoorCheck(nowRoomID)
    Beforepassagecount = FrameInfo.get_tile_count('#')
    if(GoDoorID!=-1 and GoDoorX!=-1 and GoDoorY!=-1):
        #print("There is unvisited Door! Explore Start!")
        py,px = GotoDoor(nowRoomID,GoDoorID,GoDoorY,GoDoorX)
        Passage = explore(nowRoomID,GoDoorID,Beforepassagecount,py,px)
        if(len(Passage)>0 and Passage[0]==-1):
            #print("Can't explore not visited door. End.")
            return False
        else:
            return True
    else:
        return False

def PickupAct(nowRoomID):
    PickupSearch(nowRoomID)
    if(not RoomInfoList[nowRoomID].itemList):
        return False
    itemy,itemx,obj = RoomInfoList[nowRoomID].itemList[0]
    GotoObject(nowRoomID,itemy,itemx,obj)
    return True

def ActMethod(action,nowRoomID):
    global ExtraExplore
    if(action<4):
        if(StairsFound==True):
            ExtraExplore+=1
        #print("GotoKnownRoomAct ID: {0}!".format(action))
        return GotoKnownRoomAct(nowRoomID,action),False
    elif(action==4):
        if(StairsFound==True):
            ExtraExplore+=1
        #print("Explore!")
        return ExploreAct(nowRoomID),False
    elif(action==5):
        screen=RB.get_screen()
        #print("Fight!")
        return FightAct(screen,nowRoomID),False
    elif(action==6):
        screen=RB.get_screen()
        #print("GotoStairs!")
        return GotoStairsAct(nowRoomID,screen),True
    elif(action==7):
        #print("GotoStairsRoom!")
        return GotoStairsRoomAct(nowRoomID),False
    elif(action==8):
        #print("Pickup item!")
        return PickupAct(nowRoomID),False
