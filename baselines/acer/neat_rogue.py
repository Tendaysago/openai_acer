import sys

sys.path.append("../../")
import MOneat as neat
import numpy as np
import types
from . import run_neat_base
import random
import math
import time
from time import sleep
import copy
import re
import os
import argparse
import pickle
from rogueinabox_lib.parser import RogueParser
from itertools import count, chain

RB = None
RBParser = None
FrameInfo = None
SPACE = chr(20)
ESC = chr(27)
STAR = chr(30)
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]
diagonalx = [-1, 1, 1, -1]
diagonaly = [-1, -1, 1, 1]
command = ["k", "l", "j", "h", "y", "u", "n", "b"]
atoz = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
Foodidx = -1
FoodNum = 0
ExtraExplore = 0
Moveonlynum = 0
Arrowidx = []
RoomInfoList = []
maxRoomID = -1
RightActnum = 0
ExploreActnum = 0
PrevAct = -1
Getitemnum = 0
Defeatmonsnum = 0
stepcnt = 0
VisitedRoom = None
StairsFound = False
ExploreStack = False
GlobalStackCheck = False
demoplay_mode = False
now_inventory = None
enemy_offence_mean_list = [
    2.4,
    2.6,
    3.0,
    3.5,
    4.1,
    4.4,
    5.2,
    5.7,
    6.0,
    6.6,
    6.6,
    8.5,
    8.7,
    9.5,
    10.0,
]
enemy_level_mean_list = [
    1.0,
    1.0,
    1.2,
    1.2,
    1.4,
    1.8,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    4.0,
    4.6,
    5.2,
    6.2,
]


class RoomInfo:
    def __init__(
        self,
        id,
        leftup,
        rightbottom,
        doorlist,
        knowndoorlist,
        stairexisted,
        stairsRoomdis,
        returnPassage,
    ):
        self.id = id
        self.leftup = leftup
        self.rightbottom = rightbottom
        self.doorlist = doorlist
        self.knowndoorlist = knowndoorlist
        self.stairexisted = stairexisted
        self.stairsRoomdis = stairsRoomdis
        self.itemList = []
        self.returnPassage = returnPassage


class DoorInfo:
    def __init__(self, id, Y, X, visited, passagelist, useless):
        self.id = id
        self.Y = Y
        self.X = X
        self.visited = False
        self.passagelist = passagelist
        self.useless = useless


class PassageInfo:
    def __init__(self, id, passage, connectroomid):
        self.id = id
        self.passage = passage
        self.connectroomid = connectroomid


def StairsRoomdisDFS(nowRoomID, visitedDFSArr, Prevdis):
    RoomInfoList[nowRoomID].stairsRoomdis = Prevdis
    nxtDFSvisitedArr = visitedDFSArr.copy()
    nxtDFSvisitedArr[nowRoomID] = True
    for d in range(len(RoomInfoList[nowRoomID].doorlist)):
        door = RoomInfoList[nowRoomID].doorlist[d]
        if (
            door.visited
            and (not door.useless)
            and len(RoomInfoList[nowRoomID].doorlist[d].passagelist) > 0
        ):
            nxtRoomID = door.passagelist[0].connectroomid
            if (
                not nxtDFSvisitedArr[nxtRoomID]
                and RoomInfoList[nxtRoomID].stairsRoomdis > Prevdis + 1
            ):
                StairsRoomdisDFS(nxtRoomID, nxtDFSvisitedArr, Prevdis + 1)
        else:
            continue


def RoomObjectSearch(screen, leftup, rightbottom, obj):
    ret = 0
    if obj != "monster":
        for y in range(leftup[0], rightbottom[0]):
            for x in range(leftup[1], rightbottom[1]):
                if screen[y][x] == obj:
                    ret += 1
    else:
        for y in range(leftup[0], rightbottom[0]):
            for x in range(leftup[1], rightbottom[1]):
                if screen[y][x].isupper():
                    ret += 1
    return ret


def PickupSearch(nowRoomID, playery, playerx):
    global RoomInfoList
    # room = RoomInfoList[nowRoomID]
    leftup = RoomInfoList[nowRoomID].leftup
    rightbottom = RoomInfoList[nowRoomID].rightbottom
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    RoomInfoList[nowRoomID].itemList = []
    for y in range(leftup[0], rightbottom[0]):
        for x in range(leftup[1], rightbottom[1]):
            if (
                screen[y][x] == "!"
                or screen[y][x] == "?"
                or screen[y][x] == "/"
                or screen[y][x] == ")"
                or screen[y][x] == "*"
                or screen[y][x] == ":"
                or screen[y][x] == ","
            ):
                RoomInfoList[nowRoomID].itemList.append((y, x, screen[y][x]))
    RoomInfoList[nowRoomID].itemList.sort(
        key=lambda x: math.sqrt(((playery - x[0]) ** 2) + ((playerx - x[1]) ** 2))
    )


def MonsterSearch(nowRoomID):
    global RoomInfoList
    monsternum = 0
    # room = RoomInfoList[nowRoomID]
    leftup = RoomInfoList[nowRoomID].leftup
    rightbottom = RoomInfoList[nowRoomID].rightbottom
    if RB.game_over():
        return 0
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    for y in range(leftup[0], rightbottom[0]):
        for x in range(leftup[1], rightbottom[1]):
            if screen[y][x].isupper() is True:
                monsternum += 1
    return monsternum


def CheckFoodnum():
    global now_inventory
    now_inventory = CheckInventory()
    foodnum = 0
    if "food" in now_inventory.keys():
        foodnum = now_inventory["food"]["num"]
    # print("NowFoodNum: {0}".format(RB.FoodNum))
    return foodnum


def EatFood():
    RB.pipe.write("e".encode())
    sleep(0.04)
    RB.pipe.write(STAR.encode())
    sleep(0.04)
    screen = RB.get_screen()
    Foodcommand = screen[0][0]
    RB.pipe.write(" ".encode())
    sleep(0.04)
    RB.pipe.write(Foodcommand.encode())
    RB.pipe.write(" ".encode())
    sleep(0.04)
    RB.send_command("e")


def IntelisenseItemuse(dir=-1, enemynum=0, priority="None"):
    global now_inventory
    global FrameInfo
    while wrongscreen():
        screen_refresh()
    now_inventory = CheckInventory()
    screen = RB.get_screen()
    while wrongscreen():
        screen_refresh()
    if RB.game_over():
        return False
    PCnowHP = FrameInfo.statusbar["current_hp"]
    PCmaxHP = FrameInfo.statusbar["max_hp"]
    PCnowstate = FrameInfo.statusbar["pc_status"]
    if priority == "Heal":
        if "extra" in now_inventory.keys():
            RB.send_command("q" + now_inventory["extra"]["key_val"])
            return True
        if "healng" in now_inventory.keys():
            RB.send_command("q" + now_inventory["healing"]["key_val"])
            return True
    if priority == "Overdo":
        if "teleport" in now_inventory.keys():
            RB.send_command("z" + command[dir] + now_inventory["teleport"]["key_val"])
            return True
    if priority == "Weak":
        if "slow" in now_inventory.keys():
            RB.send_command("z" + command[dir] + now_inventory["slow"]["key_val"])
            return True
    if priority == "Powerup":
        if "haste" in now_inventory.keys() and PCnowstate == "Normal":
            RB.send_command("q" + now_inventory["haste"]["key_val"])
            return True
    if priority == "Use":
        canuse = False
        if "healing" in now_inventory.keys():
            if now_inventory["healing"]["num"] >= 3 and PCnowHP == PCmaxHP:
                RB.send_command("q" + now_inventory["healing"]["key_val"])
                canuse = True
        if "extra" in now_inventory.keys():
            if now_inventory["extra"]["num"] >= 4 and PCnowHP == PCmaxHP:
                RB.send_command("q" + now_inventory["extra"]["key_val"])
                canuse = True
        if "strength" in now_inventory.keys():
            RB.send_command("q" + now_inventory["strength"]["key_val"])
            canuse = True
        if "weapon" in now_inventory.keys():
            RB.send_command("r" + now_inventory["weapon"]["key_val"])
            canuse = True
        if "armor" in now_inventory.keys():
            RB.send_command("r" + now_inventory["armor"]["key_val"])
            canuse = True
        if "food" in now_inventory.keys():
            if now_inventory["food"]["num"] >= 5:
                RB.send_command("e" + now_inventory["food"]["key_val"])
                canuse = True
        return canuse
    if priority == "Eat":
        canuse = False
        if "Food" in now_inventory.keys():
            RB.send_command("e" + now_inventory["Food"]["key_val"])
            canuse = True
        return canuse

    return False


def IntelisenseFight(enemy_dir, diagonal=False):
    global FrameInfo
    if RB.game_over():
        return RB.game_over()
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY, PlayerX = FrameInfo.get_player_pos()
    PlayerY += 1
    if demoplay_mode is True:
        Screenprint(screen)
        sleep(0.05)
    enenum = Approachenemynum(PlayerY, PlayerX, diagonal)
    PCnowHP = FrameInfo.statusbar["current_hp"]
    PCmaxHP = FrameInfo.statusbar["max_hp"]
    if enenum >= 2:
        if IntelisenseItemuse(enemy_dir, enenum, "Overdo") is False:
            if IntelisenseItemuse(enemy_dir, enenum, "Powerup") is False:
                RB.send_command(command[enemy_dir])
        if PCnowHP <= (PCmaxHP / 2):
            IntelisenseItemuse(-1, 0, priority="Heal")
    elif PCnowHP <= (PCmaxHP / 2):
        IntelisenseItemuse(-1, 0, priority="Heal")
        RB.send_command(command[enemy_dir])
    else:
        if PCnowHP <= (PCmaxHP / 2):
            IntelisenseItemuse(-1, 0, priority="Heal")
        print("Attack command : {0}".format(command[enemy_dir]))
        RB.send_command(command[enemy_dir])
    screen = RB.get_screen()
    if demoplay_mode is True:
        Screenprint(screen)
        sleep(0.05)
    return RB.game_over()


def CloseCombatinpassage():
    global FrameInfo
    while True:
        if RB.game_over():
            return RB.game_over()
        while wrongscreen():
            screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        fightcontinue = False
        for dir in range(4):
            if screen[PlayerY + dy[dir]][PlayerX + dx[dir]].isupper():
                print("Combat in Passage!")
                IntelisenseFight(dir)
                fightcontinue = True
                break
        if fightcontinue is False or RB.game_over():
            break
    return RB.game_over()


def CloseCombatinroom(nowRoomID):
    global now_inventory
    if RB.game_over():
        return RB.game_over()
    while wrongscreen():
        screen_refresh()
    if MonsterSearch(nowRoomID) == 0:
        return RB.game_over()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY, PlayerX = FrameInfo.get_player_pos()
    PlayerY += 1

    doorretreat = -1
    for dir in range(4):
        if screen[PlayerY + dy[dir]][PlayerX + dx[dir]] == "+":
            RB.send_command(command[dir])
            doorretreat = (dir + 2) % 4
            break
    if RB.game_over():
        return RB.game_over()
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY, PlayerX = FrameInfo.get_player_pos()
    PlayerY += 1
    if doorretreat != -1 or FrameInfo.get_tile_below_player() == "+":
        while MonsterSearch(nowRoomID) > 0 or Approachenemynum(PlayerY, PlayerX) > 0:
            if RB.game_over():
                return RB.game_over()
            print("Combat in Room! (on door)")
            while Approachenemynum(PlayerY, PlayerX) == 0:
                if RB.game_over():
                    return RB.game_over()
                while wrongscreen():
                    screen_refresh()
                screen = RB.get_screen()
                RB.send_command(".")
                if demoplay_mode is True:
                    Screenprint(screen)
                    sleep(0.05)
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            if demoplay_mode is True:
                Screenprint(screen)
                sleep(0.05)
            fightcontinue = False
            for dir in range(4):
                if screen[PlayerY + dy[dir]][PlayerX + dx[dir]].isupper():
                    print("Attack!")
                    if demoplay_mode is True:
                        Screenprint(screen)
                        sleep(0.05)
                    IntelisenseFight(dir)
                    fightcontinue = True
                    break
            if fightcontinue is False or RB.game_over():
                break
            if RB.game_over is True:
                break
    else:
        while (
            MonsterSearch(nowRoomID) > 0
            or Approachenemynum(PlayerY, PlayerX, diagonal=True) > 0
        ):
            print(MonsterSearch(nowRoomID))
            print(Approachenemynum(PlayerY, PlayerX))
            if RB.game_over():
                return RB.game_over()
            print("Combat in Room!")
            while Approachenemynum(PlayerY, PlayerX, diagonal=True) == 0:
                if RB.game_over():
                    return RB.game_over()
                while wrongscreen():
                    screen_refresh()
                screen = RB.get_screen()
                RB.send_command(".")
                if demoplay_mode is True:
                    Screenprint(screen)
                    sleep(0.05)
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            if demoplay_mode is True:
                Screenprint(screen)
                sleep(0.05)
            fightcontinue = False
            for dir in range(4):
                if screen[PlayerY + dy[dir]][PlayerX + dx[dir]].isupper():
                    print("Attack!")
                    if demoplay_mode is True:
                        Screenprint(screen)
                        sleep(0.05)
                    IntelisenseFight(dir)
                    fightcontinue = True
                if (
                    screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != "-"
                    and screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != "|"
                ):
                    for diadir in range(2):
                        if screen[PlayerY + diagonaly[(dir + diadir) % 4]][
                            PlayerX + diagonalx[(dir + diadir) % 4]
                        ].isupper():
                            print("Attack!")
                            if demoplay_mode is True:
                                Screenprint(screen)
                                sleep(0.05)
                            if dir + diadir + 4 < 8:
                                IntelisenseFight(dir + diadir + 4)
                            else:
                                IntelisenseFight(dir + diadir)
                            fightcontinue = True
            if fightcontinue is False or RB.game_over():
                break
            if RB.game_over is True:
                break
    if doorretreat != -1:
        RB.send_command(command[doorretreat])

    return RB.game_over()


def Approachenemynum(py, px, diagonal=False):
    enemynum = 0
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    for dir in range(4):
        if screen[py + dy[dir]][px + dx[dir]].isupper():
            enemynum += 1
        if (
            diagonal is True
            and screen[py + dy[dir]][px + dx[dir]] != "-"
            and screen[py + dy[dir]][px + dx[dir]] != "|"
        ):
            for diadir in range(2):
                if screen[py + diagonaly[(dir + diadir) % 4]][
                    px + diagonalx[(dir + diadir) % 4]
                ].isupper():
                    enemynum += 1
    return enemynum


def CheckInventory():
    return RB.send_command("i", check_inventory=True)


def MakeInput(nowRoomID, t):
    global FrameInfo
    global FoodNum
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    # input = np.full(24,-1.0,dtype=float)
    input = np.full(10, -1.0, dtype=float)
    if StairsFound:
        input[0] = 1.0
    Room = RoomInfoList[nowRoomID]
    objnum = 0
    i = 0
    input[3] = 100
    for d in range(len(Room.doorlist)):
        if not Room.doorlist[d].visited:
            input[1] = 1.0
        if Room.doorlist[d].useless == False and len(Room.doorlist[d].passagelist) > 0:
            nxtroom = RoomInfoList[Room.doorlist[d].passagelist[0].connectroomid]
            for nd in range(len(nxtroom.doorlist)):
                if not nxtroom.doorlist[nd].visited:
                    input[2] = 1.0
            if nxtroom.stairsRoomdis != 100:
                input[3] = min(input[3], nxtroom.stairsRoomdis / 6.0)
    if input[3] == 100:
        input[3] = 1.0
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, ":")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "!")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "?")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, ")")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "*")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "/")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, ",")
    input[4] = 1 if objnum > 0 else 0
    input[5] = CheckFoodnum()
    FoodNum = input[5]
    if input[5] == 1:
        input[5] = 0.5
    elif input[5] > 1:
        input[5] = 1.0
    input[6] = (20.0 - t) / 20.0
    input[7] = Getitemnum / 5.0
    input[8] = maxRoomID / 5.0
    if maxRoomID == 0:
        input[9] = 0
    else:
        input[9] = 1
    return input


def MakecommanderInput(nowRoomID, t, prevmodular):
    global FrameInfo
    global FoodNum
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    while wrongscreen():
        screen_refresh()
    input = np.full(5, -1.0, dtype=float)
    if StairsFound:
        input[0] = 1.0
    input[1] = FrameInfo.statusbar["exp_level"] / 15.0
    input[2] = FrameInfo.statusbar["dungeon_level"] / 15.0
    inventory = CheckInventory()
    for key, val in inventory:
        input[3] += val
    input[3] /= 20.0
    input[4] = (
        enemy_offence_mean_list[(FrameInfo.statusbar["dungeon_level"] - 1)] / 10.0
    )
    return input


def RoominfoPrint(nowRoomID):
    print("RoomInformation print!")
    print("Now Room ID: {0}".format(nowRoomID))
    print(
        "LeftUp Y {0}, X{1},".format(
            RoomInfoList[nowRoomID].leftup[0], RoomInfoList[nowRoomID].leftup[1]
        )
    )
    print(
        "RightBottom Y {0}, X{1},".format(
            RoomInfoList[nowRoomID].rightbottom[0],
            RoomInfoList[nowRoomID].rightbottom[1],
        )
    )
    Room = RoomInfoList[nowRoomID]
    PickupSearch(nowRoomID)
    for item in RoomInfoList[nowRoomID].itemList:
        print(item)
    for d in range(len(Room.doorlist)):
        door = Room.doorlist[d]
        print(
            "Door ID:{0}, Coordinates:({1},{2}), Visited:{3}, Useless:{4}.".format(
                door.id, door.Y, door.X, door.visited, door.useless
            )
        )
        for p in range(len(door.passagelist)):
            Passage = door.passagelist[p]
            route = copy.deepcopy(Passage.passage)
            for tmp in range(len(route)):
                if route[tmp] == 0:
                    route[tmp] = "↑"
                elif route[tmp] == 1:
                    route[tmp] = "→"
                elif route[tmp] == 2:
                    route[tmp] = "↓"
                elif route[tmp] == 3:
                    route[tmp] = "←"
            print(
                "Passage ID:{0}, passage:{1}, connectRoomID:{2}".format(
                    Passage.id, route, Passage.connectroomid
                )
            )
    objnum = 0
    screen = RB.get_screen()
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, ":")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "!")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "?")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, ")")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "monster")
    objnum += RoomObjectSearch(screen, Room.leftup, Room.rightbottom, "*")
    print("Room objectnum:{0}".format(objnum))
    print("StairsExisted:{0}".format(Room.stairexisted))
    Screenprint(screen)


def StairsCheck(nowRoomID):
    global FrameInfo
    global StairsFound
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)

    if not StairsFound and FrameInfo.get_list_of_positions_by_tile("%") != []:
        StairsFound = True
        RoomInfoList[nowRoomID].stairexisted = True
        RoomInfoList[nowRoomID].stairsRoomdis = 0
        DFSArr = np.array([False] * 15)
        StairsRoomdisDFS(nowRoomID, DFSArr, 0)


def Screenprint(screen):
    i = 0
    for i in range(24):
        sys.stdout.write("\r{0}\n".format(screen[i]))
        sys.stdout.flush()
    print("\033[24A", end="")
    # print(screen[i])
    # sys.stdout.flush()
    time.sleep(0.005)


def RoomInfoMake(PlayerY, PlayerX, screen, nowRoomID, returnPassage):
    global RoomInfoList
    global VisitedRoom
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    if VisitedRoom[nowRoomID] == False:
        # print("Room id {0} is New Room! Get LeftUp and RightBottom and doorlist!".format(nowRoomID))
        RoomInfoList[nowRoomID].leftup = getleftup(PlayerY, PlayerX, screen)
        RoomInfoList[nowRoomID].rightbottom = getrightbottom(PlayerY, PlayerX, screen)
        RoomInfoList[nowRoomID].doorlist = getdoorList(
            RoomInfoList[nowRoomID].leftup[0], RoomInfoList[nowRoomID].leftup[1], screen
        )
        VisitedRoom[nowRoomID] = True
        RoomInfoList[nowRoomID].returnPassage = returnPassage
    else:
        pass  # print("Room id {0} is Not New Room. Not need make New Room Info.".format(nowRoomID))


def StairsRoomdisCheck(nowRoomID, Prevdis):
    if Prevdis == 100:
        return
    RoomInfoList[nowRoomID].stairsRoomdis = min(
        RoomInfoList[nowRoomID].stairsRoomdis, Prevdis + 1
    )


def RightMethod(playery, playerx):
    global FrameInfo
    global ExploreStack
    global GlobalStackCheck
    passage = []
    prevdir = -1
    CanNext = False
    CanThrough = False
    if RB.game_over():
        return [], -1, -1, -1
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY, PlayerX = playery, playerx
    # PlayerY+=1
    stackcheck = 0
    retrycnt = 0
    stack = False
    # if(FrameInfo.get_tile_below_player()!='+'):
    #     print("Warning! below_player tile is not door!")
    # 扉からの最初の一歩
    dir = 0
    i = 0
    while passage == [] and FrameInfo.get_tile_below_player() == "+":
        if RB.game_over():
            return [], -1, -1, -1
        while wrongscreen():
            screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        i += 1
        if i % 100 == 0:
            print("Warning! below_player tile is not door!")
            Screenprint(screen)
        if RB.game_over():
            return [], -1, -1, -1
        for dir in range(4):
            if (
                screen[PlayerY + dy[dir]][PlayerX + dx[dir]] == "#"
                or screen[PlayerY + dy[dir]][PlayerX + dx[dir]] == " "
            ):
                RB.send_command(command[dir])
                screen = RB.get_screen()
                FrameInfo = RBParser.parse_screen(screen)
                prevdir = dir
                passage.append(dir)
                break
            elif screen[PlayerY + dy[dir]][PlayerX + dx[dir]].isupper():
                # RB.send_command(command[dir])
                CloseCombatinpassage()
                screen = RB.get_screen()
                FrameInfo = RBParser.parse_screen(screen)
                prevdir = dir
                # Screenprint(screen)
                if RB.game_over():
                    return [], -1, -1, -1
                # debug_print(0,0)
                FrameInfo = RBParser.parse_screen(screen)
                break
    # for i in range(24):
    #    print(screen[i])

    # print("Right Method Start!")
    if prevdir == -1:
        print("Warning! prevdir=-1!?")
        print(passage, prevdir)
        Screenprint(screen)
        debug_print(0, 0)
        for dir in range(4):
            if (
                screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != "-"
                and screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != "|"
                and screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != "#"
                and screen[PlayerY + dy[dir]][PlayerX + dx[dir]] != " "
            ):
                RB.send_command(command[(dir + 2) % 4])
                prevdir = (dir + 2) % 4
                passage.append((dir + 2) % 4)
                screen = RB.get_screen()
                # Screenprint(screen)
                break
    while True:
        if RB.game_over() == True:
            return [], -1, -1, -1
        if stackcheck >= 200 and stack == False:
            stack = True
            print("Warning, Maybe Stack.")
            screen = RB.get_screen()
            # Screenprint(screen)
            # print(passage)
            ExploreStack = True
            GlobalStackCheck = True
            break
        stackcheck += 1
        CanNext = False
        CanContinue = False
        """
        screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        """
        if RB.game_over() is True:
            return [], -1, -1, -1
        # Screenprint(screen)
        while wrongscreen():
            screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.05)
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        # 進める限り右手法で進もうとする。
        # for i in range(24):
        #    print(screen[i])
        # まずは進行方向の右側が空いているか?
        rightcommand = (prevdir + 1) % 4
        frontcommand = prevdir
        leftcommand = (prevdir + 3) % 4
        BeforeY = PlayerY
        BeforeX = PlayerX
        # sleep(0.2)
        if screen[PlayerY + dy[rightcommand]][PlayerX + dx[rightcommand]].isupper():
            CanNext = True
            CloseCombatinpassage()
            # RB.send_command(command[rightcommand])
            continue
        RB.send_command(command[rightcommand])
        """
        screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        """
        if RB.game_over():
            return [], -1, -1, -1
        while wrongscreen():
            screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        AfterY, AfterX = FrameInfo.get_player_pos()
        AfterY += 1
        if AfterY == (BeforeY + dy[rightcommand]) and AfterX == (
            BeforeX + dx[rightcommand]
        ):
            prevdir = (rightcommand) % 4
            passage.append(rightcommand)
            retrycnt = 0
            if (
                FrameInfo.get_tile_below_player() == "#"
                or FrameInfo.get_tile_below_player() == " "
            ):
                CanNext = True
            elif FrameInfo.get_tile_below_player() == "+":
                CanThrough = True
            continue
        if CanNext == False and CanThrough == False:
            if screen[PlayerY + dy[frontcommand]][PlayerX + dx[frontcommand]].isupper():
                CanNext = True
                CloseCombatinpassage()
                # RB.send_command(command[frontcommand])
                continue
            RB.send_command(command[frontcommand])
            """
            screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            """
            if RB.game_over():
                return [], -1, -1, -1
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            AfterY, AfterX = FrameInfo.get_player_pos()
            AfterY += 1
            # print(FrameInfo.get_tile_below_player())
            if AfterY == (BeforeY + dy[frontcommand]) and AfterX == (
                BeforeX + dx[frontcommand]
            ):
                prevdir = frontcommand % 4
                passage.append(frontcommand)
                retrycnt = 0
                if (
                    FrameInfo.get_tile_below_player() == "#"
                    or FrameInfo.get_tile_below_player() == " "
                ):
                    CanNext = True
                elif FrameInfo.get_tile_below_player() == "+":
                    CanThrough = True
                continue
        if CanNext == False and CanThrough == False:
            if screen[PlayerY + dy[leftcommand]][PlayerX + dx[leftcommand]].isupper():
                CanNext = True
                CloseCombatinpassage()
                # RB.send_command(command[leftcommand])
                continue
            RB.send_command(command[leftcommand])
            """
            screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            """
            if RB.game_over():
                return [], -1, -1, -1
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            AfterY, AfterX = FrameInfo.get_player_pos()
            AfterY += 1
            # print(FrameInfo.get_tile_below_player())
            if AfterY == (BeforeY + dy[leftcommand]) and AfterX == (
                BeforeX + dx[leftcommand]
            ):
                prevdir = (leftcommand) % 4
                passage.append(leftcommand)
                retrycnt = 0
                if (
                    FrameInfo.get_tile_below_player() == "#"
                    or FrameInfo.get_tile_below_player() == " "
                ):
                    CanNext = True
                elif FrameInfo.get_tile_below_player() == "+":
                    CanThrough = True
                continue
        if CanThrough:
            # print("CanThrough!")
            """
            screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            """
            # Screenprint(screen)
            # print("Passage: {0}".format(passage))
            # print("-----------")
            if RB.game_over():
                return [], -1, -1, -1
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            AfterY, AfterX = FrameInfo.get_player_pos()
            AfterY += 1
            # RB.send_command(command[prevdir])
            screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            if RB.game_over():
                return [], -1, -1
            while wrongscreen():
                screen_refresh()
            # LastY, LastX = FrameInfo.get_player_pos()
            LastY, LastX = AfterY + dy[prevdir], AfterX + dx[prevdir]
            # print("And throwing door to enter room.")
            return passage, AfterY, AfterX, prevdir
            while LastY == AfterY and LastX == AfterX:
                print(
                    "LastY:{0}, LastX:{1}, AfterY:{2}, AfterY:{3}".format(
                        LastY, LastX, AfterY, AfterX
                    )
                )
                screen = RB.get_screen()
                FrameInfo = RBParser.parse_screen(screen)
                if RB.game_over():
                    return [], -1, -1, -1
                while wrongscreen():
                    screen_refresh()
                RB.send_command(command[prevdir])
                screen = RB.get_screen()
                # Screenprint(screen)
                FrameInfo = RBParser.parse_screen(screen)
                if RB.game_over():
                    return [], -1, -1, -1
                while wrongscreen():
                    screen_refresh()
                LastY, LastX = FrameInfo.get_player_pos()
                LastY += 1
            break
        # これ以上通路も扉もなく行き止まりに当たったら...
        if CanNext == False and CanThrough == False:
            passage.reverse()
            i = 0
            retrycnt += 1
            if retrycnt < 3:
                passage.reverse()
                screen_refresh()
                continue
            while i < len(passage):
                screen = RB.get_screen()
                if screen[PlayerY + dy[(passage[i] + 2) % 4]][
                    PlayerX + dx[(passage[i] + 2) % 4]
                ].isupper():
                    print("これ以上進めない.")
                    debug_print(0, 0)
                    # Screenprint(screen)
                    RB.send_command(command[(passage[i] + 2) % 4])
                else:
                    RB.send_command(
                        command[(passage[i] + 2) % 4]
                    )  # 通ってきた通路を引き返し,戻っていくが...ここでも戻りつつ右手法を試みる。
                    prevdir = (passage[i] + 2) % 4
                    screen = RB.get_screen()
                    rightcommand = (prevdir + 1) % 4
                    frontcommand = prevdir
                    leftcommand = (prevdir + 3) % 4
                    # (screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='#' or screen[PlayerY+dy[frontcommand]][PlayerX+dx[frontcommand]]=='+') or\
                    # ↓引き返す時に右手法で行けるところを見つけたら、whileループの開始のところまで戻る。
                    if (
                        (
                            screen[PlayerY + dy[rightcommand]][
                                PlayerX + dx[rightcommand]
                            ]
                            == "#"
                            or screen[PlayerY + dy[rightcommand]][
                                PlayerX + dx[rightcommand]
                            ]
                            == "+"
                            or screen[PlayerY + dy[rightcommand]][
                                PlayerX + dx[rightcommand]
                            ].isupper()
                        )
                        or (
                            screen[PlayerY + dy[frontcommand]][
                                PlayerX + dx[frontcommand]
                            ]
                            == "#"
                            or screen[PlayerY + dy[frontcommand]][
                                PlayerX + dx[frontcommand]
                            ]
                            == "+"
                            or screen[PlayerY + dy[frontcommand]][
                                PlayerX + dx[frontcommand]
                            ].isupper()
                        )
                        or (
                            screen[PlayerY + dy[leftcommand]][PlayerX + dx[leftcommand]]
                            == "#"
                            or screen[PlayerY + dy[leftcommand]][
                                PlayerX + dx[leftcommand]
                            ]
                            == "+"
                            or screen[PlayerY + dy[leftcommand]][
                                PlayerX + dx[leftcommand]
                            ].isupper()
                        )
                    ):
                        CanContinue = True
                        del passage[: i + 1]
                        passage.reverse()
                        break
                    i += 1

            if CanContinue:
                continue
            if len(passage) > 0:  # 扉から一歩以上は進めたが行き止まりだった場合
                RB.send_command(command[(passage[len(passage) - 1] + 2) % 4])
            else:  # 扉に入っていきなり行き止まりだった場合
                screen = RB.get_screen()
                for i in range(4):
                    if (
                        screen[PlayerY + dy[i]][PlayerX + dx[i]] != " "
                        or screen[PlayerY + dy[i]][PlayerX + dx[i]] != "|"
                        or screen[PlayerY + dy[i]][PlayerX + dx[i]] != "-"
                    ):
                        RB.send_command(command[i])
                        break
            passage = []
            break

    # print("Right Method End!")
    # for i in range(24):
    #    print(screen[i])
    while wrongscreen():
        screen_refresh()
    return passage, AfterY, AfterX


def getleftup(y, x, screen):
    nowy = y
    nowx = x
    # print(screen[nowy][nowx])
    while (
        nowy > 0
        and nowx > 0
        and screen[nowy][nowx] != "|"
        and screen[nowy][nowx] != "-"
        and screen[nowy][nowx] != "+"
        and screen[nowy][nowx] != "#"
        and screen[nowy - 1][nowx - 1] != "#"
        and screen[nowy - 1][nowx - 1] != " "
    ):
        nowy -= 1
        nowx -= 1
    # print(nowy, nowx)
    # print(screen[nowy][nowx])
    if screen[nowy][nowx] == "|":
        while (
            nowy > 0
            and (
                screen[nowy][nowx] == "|"
                or screen[nowy][nowx] == "+"
                or screen[nowy][nowx] == "@"
                or screen[nowy][nowx].isupper()
            )
            and screen[nowy][nowx + 1] != "-"
            and screen[nowy][nowx + 1] != "+"
        ):
            nowy -= 1
            # print(nowy)
    elif screen[nowy][nowx] == "-":
        while (
            nowx > 0
            and screen[nowy][nowx - 1] != " "
            and screen[nowy][nowx - 1] != "#"
            and screen[nowy + 1][nowx] != "|"
            and screen[nowy + 1][nowx] != "+"
        ):
            nowx -= 1
            # print(nowx)
    elif screen[nowy][nowx] == "+" or screen[nowy][nowx].isupper():
        if screen[nowy][nowx - 1] == "-":
            while (
                nowx > 0
                and screen[nowy][nowx - 1] != " "
                and screen[nowy][nowx - 1] != "#"
                and screen[nowy + 1][nowx] != "|"
                and screen[nowy + 1][nowx] != "+"
            ):
                nowx -= 1
                # print(nowx)
        elif screen[nowy - 1][nowx] != " ":
            while (
                nowy > 0
                and screen[nowy - 1][nowx] != " "
                and screen[nowy - 1][nowx] != "#"
                and screen[nowy][nowx + 1] != "-"
                and screen[nowy][nowx + 1] != "+"
            ):
                nowy -= 1
                # print(nowy)
    # print(nowy,nowx)
    return nowy, nowx


def getrightbottom(y, x, screen):
    nowy = y
    nowx = x
    while (
        nowx < 80
        and nowy < 22
        and screen[nowy][nowx] != "|"
        and screen[nowy][nowx] != "-"
        and screen[nowy][nowx] != "+"
        and screen[nowy][nowx] != "#"
        and screen[nowy + 1][nowx + 1] != "#"
        and screen[nowy + 1][nowx + 1] != " "
    ):
        nowy += 1
        nowx += 1
    if screen[nowy][nowx] == "|":
        while (
            nowy < 22
            and (
                screen[nowy][nowx] == "|"
                or screen[nowy][nowx] == "+"
                or screen[nowy][nowx] == "@"
                or screen[nowy][nowx].isupper()
            )
            and screen[nowy][nowx - 1] != "-"
            and screen[nowy][nowx - 1] != "+"
        ):
            nowy += 1
    elif screen[nowy][nowx] == "-":
        while (
            nowx < 80
            and screen[nowy][nowx + 1] != " "
            and screen[nowy][nowx + 1] != "#"
            and screen[nowy - 1][nowx] != "|"
            and screen[nowy - 1][nowx] != "+"
        ):
            nowx += 1
    elif screen[nowy][nowx] == "+" or screen[nowy][nowx].isupper():
        if screen[nowy][nowx + 1] == "-":
            while (
                nowx < 80
                and screen[nowy][nowx + 1] != " "
                and screen[nowy][nowx + 1] != "#"
                and screen[nowy - 1][nowx] != "|"
                and screen[nowy - 1][nowx] != "+"
            ):
                nowx += 1
        elif screen[nowy + 1][nowx] != " ":
            while (
                nowy < 22
                and screen[nowy + 1][nowx] != " "
                and screen[nowy + 1][nowx] != "#"
                and screen[nowy][nowx - 1] != "-"
                and screen[nowy][nowx - 1] != "+"
            ):
                nowy += 1
    return nowy, nowx


# 部屋のドア一覧を獲得する関数。
# 左上から右→下→左→上と一周しながら見つけた順にドア情報をリストに挿入し、最後にドアリストを返す。
def getdoorList(LeftupY, LeftupX, screen):
    nowy = LeftupY
    nowx = LeftupX
    doorList = []
    id = 0
    # まずは右へ
    while (
        nowx <= 80 and screen[nowy][nowx + 1] != " " and screen[nowy][nowx + 1] != "#"
    ):
        nowx += 1
        if (
            screen[nowy][nowx] == "+"
            or screen[nowy][nowx] == "@"
            or screen[nowy][nowx].isupper()
        ):
            doorList.append(DoorInfo(id, nowy, nowx, False, [], False))
            id += 1
        if screen[nowy + 1][nowx] == "|" or screen[nowy + 1][nowx] == "+":
            break
        # nowx+=1
    # 下へ
    while (
        nowy <= 22 and screen[nowy + 1][nowx] != " " and screen[nowy + 1][nowx] != "#"
    ):
        nowy += 1
        if (
            screen[nowy][nowx] == "+"
            or screen[nowy][nowx] == "@"
            or screen[nowy][nowx].isupper()
        ):
            doorList.append(DoorInfo(id, nowy, nowx, False, [], False))
            id += 1
        if screen[nowy][nowx - 1] == "-" or screen[nowy][nowx - 1] == "+":
            break
        # nowy+=1
    # 左へ
    while nowx >= 0 and screen[nowy][nowx - 1] != " " and screen[nowy][nowx - 1] != "#":
        nowx -= 1
        if (
            screen[nowy][nowx] == "+"
            or screen[nowy][nowx] == "@"
            or screen[nowy][nowx].isupper()
        ):
            doorList.append(DoorInfo(id, nowy, nowx, False, [], False))
            id += 1
        if screen[nowy - 1][nowx] == "|" or screen[nowy - 1][nowx] == "+":
            break
        # nowx-=1
    # 上へ
    while nowy >= 1 and screen[nowy - 1][nowx] != " " and screen[nowy - 1][nowx] != "#":
        nowy -= 1
        if (
            screen[nowy][nowx] == "+"
            or screen[nowy][nowx] == "@"
            or screen[nowy][nowx].isupper()
        ):
            doorList.append(DoorInfo(id, nowy, nowx, False, [], False))
            id += 1
        if screen[nowy][nowx + 1] == "-" or screen[nowy][nowx + 1] == "+":
            break
        # nowy-=1
    return doorList


def NotvisitedDoorCheck(nowRoomID, playery, playerx):
    global RoomInfoList
    mindist = 1000
    goDoorid = -1
    DoorX = -1
    DoorY = -1
    for i in range(len(RoomInfoList[nowRoomID].doorlist)):
        doordist = math.sqrt(
            (RoomInfoList[nowRoomID].doorlist[i].Y - playery) ** 2
            + (RoomInfoList[nowRoomID].doorlist[i].X - playerx) ** 2
        )
        if RoomInfoList[nowRoomID].doorlist[i].visited is False and doordist < mindist:
            # print("Door ID" + str(RoomInfoList[nowRoomID].doorlist[i].id) + " not Visited!")
            doordist = mindist
            goDoorid = i
            DoorX = RoomInfoList[nowRoomID].doorlist[i].X
            DoorY = RoomInfoList[nowRoomID].doorlist[i].Y
    return goDoorid, DoorX, DoorY


def GotoObject(nowRoomID, aimY, aimX, object):
    global GlobalStackCheck
    global Getitemnum
    global FrameInfo
    global now_inventory
    if object == "%" and RoomInfoList[nowRoomID].stairexisted == False:
        return False
    # 階段のところまで地道にいき、階段にたどり着いたら階段を降りる。
    stackcheck = 0
    stack = False
    if object == "%":
        aimY += 1
    PlayerY = -1
    PlayerX = -1
    while True:
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if RB.game_over():
            return False
        while wrongscreen():
            screen_refresh()
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.05)
        CloseCombatinroom(nowRoomID)
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        if PlayerY != aimY or PlayerX != aimX:
            stackcheck += 1
            if (
                PlayerX < aimX
                and PlayerY < aimY
                and screen[PlayerY + 1][PlayerX + 1] != "|"
                and screen[PlayerY + 1][PlayerX + 1] != "-"
                and screen[PlayerY + 1][PlayerX + 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY + 1][PlayerX + 1].isupper():
                    RB.send_command("n")
                    continue
                RB.send_command("n")
                PlayerX += 1
                PlayerY += 1
            elif (
                PlayerX > aimX
                and PlayerY < aimY
                and screen[PlayerY + 1][PlayerX - 1] != "|"
                and screen[PlayerY + 1][PlayerX - 1] != "-"
                and screen[PlayerY + 1][PlayerX - 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY + 1][PlayerX - 1].isupper():
                    RB.send_command("b")
                    continue
                RB.send_command("b")
                PlayerX -= 1
                PlayerY += 1
            elif (
                PlayerX < aimX
                and PlayerY > aimY
                and screen[PlayerY - 1][PlayerX + 1] != "|"
                and screen[PlayerY - 1][PlayerX + 1] != "-"
                and screen[PlayerY - 1][PlayerX + 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY - 1][PlayerX + 1].isupper():
                    RB.send_command("u")
                    continue
                RB.send_command("u")
                PlayerX += 1
                PlayerY -= 1
            elif (
                PlayerX > aimX
                and PlayerY > aimY
                and screen[PlayerY - 1][PlayerX - 1] != "|"
                and screen[PlayerY - 1][PlayerX - 1] != "-"
                and screen[PlayerY - 1][PlayerX - 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY - 1][PlayerX - 1].isupper():
                    RB.send_command("y")
                    continue
                RB.send_command("y")
                PlayerX -= 1
                PlayerY -= 1
            elif (
                PlayerX < aimX
                and screen[PlayerY][PlayerX + 1] != "|"
                and screen[PlayerY][PlayerX + 1] != "-"
            ):
                if screen[PlayerY][PlayerX + 1].isupper():
                    RB.send_command("l")
                    continue
                RB.send_command("l")
                PlayerX += 1
            elif (
                PlayerX > aimX
                and screen[PlayerY][PlayerX - 1] != "|"
                and screen[PlayerY][PlayerX - 1] != "-"
            ):
                if screen[PlayerY][PlayerX - 1].isupper():
                    RB.send_command("h")
                    continue
                RB.send_command("h")
                PlayerX -= 1
            elif (
                PlayerY < aimY
                and screen[PlayerY + 1][PlayerX] != "-"
                and screen[PlayerY + 1][PlayerX] != "|"
            ):
                if screen[PlayerY + 1][PlayerX].isupper():
                    RB.send_command("j")
                    continue
                RB.send_command("j")
                PlayerY += 1
            elif (
                PlayerY > aimY
                and screen[PlayerY - 1][PlayerX] != "-"
                and screen[PlayerY - 1][PlayerX] != "|"
            ):
                if screen[PlayerY - 1][PlayerX].isupper():
                    RB.send_command("k")
                    continue
                RB.send_command("k")
                PlayerY -= 1
            if stackcheck >= 100 and stack == False:
                stack = True
                # print("Warning, Maybe Stack while go to stairs.")
                # debug_print(nowRoomID,0)
                # screen = RB.get_screen()
                # Screenprint(screen)
                # print("DoorList")
                # print("NowRoomID: {0}".format(nowRoomID))
                # print("Now Player:({0},{1})".format(RB.player_pos[0],RB.player_pos[1]))
                # print("Stairs:({0},{1})".format(aimY,aimX))
                # for ridx in range(len(RoomInfoList)):
                #     if(VisitedRoom[ridx]):
                #         print("RoomID: {0}".format(ridx))
                #         print("LeftUp Y {0}, X{1},".format(RoomInfoList[ridx].leftup[0],RoomInfoList[ridx].leftup[1]))
                #         print("RightBottom Y {0}, X{1},".format(RoomInfoList[ridx].rightbottom[0],RoomInfoList[ridx].rightbottom[1]))
                #         print("Stairs Existed{0}".format(RoomInfoList[ridx].stairexisted))
                #         i=0
                #         for i in range(len(RoomInfoList[ridx].doorlist)):
                #             DoorID= RoomInfoList[ridx].doorlist[i].id
                #             DoorX = RoomInfoList[ridx].doorlist[i].X
                #             DoorY = RoomInfoList[ridx].doorlist[i].Y
                #             CanThrough = len(RoomInfoList[ridx].doorlist[i].passagelist)==0 and RoomInfoList[ridx].doorlist[i].visited
                #             print(DoorID,DoorY,DoorX,CanThrough)
                GlobalStackCheck = True
                break
        else:
            break
    while wrongscreen():
        screen_refresh()
    """
    if FrameInfo.get_player_pos() == None:
        # screen=RB.get_screen()
        screen_refresh()
        # Screenprint(screen)
        # RB.send_command('i')
        # sleep(RB.busy_wait_seconds*200)
        screen = RB.get_screen()
        # Screenprint(screen)
        return False
    """
    if object == "%":
        RB.send_command(">")
    else:
        Getitemnum += 1
        RB.send_command(",")
        now_inventory = CheckInventory()
        # print(now_inventory)
        IntelisenseItemuse(dir=-1, enemynum=0, priority="Use")
    return True


def GotoDoor(nowRoomID, goDoorid, DoorY, DoorX):
    global GlobalStackCheck
    global FrameInfo
    RoomInfoList[nowRoomID].doorlist[goDoorid].visited = True
    screen = RB.get_screen()
    PlayerY = -1
    PlayerX = -1
    # まずは扉の所まで地道にいく。
    # 部屋の左上右下の取得,扉リストの取得がおかしいとここでループが終わらない。(壁にガンガン)
    stackcheck = 0
    stack = False
    while True:
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if RB.game_over():
            return -1, -1
        while wrongscreen():
            screen_refresh()
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.05)
        screen = RB.get_screen()
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        CloseCombatinroom(nowRoomID)
        # print("Player(Y,X):({0},{1})".format(PlayerY,PlayerX))
        # Screenprint(screen)
        if PlayerY != DoorY or PlayerX != DoorX:
            if RB.game_over() is True:
                return -1, -1
            stackcheck += 1
            if (
                PlayerX < DoorX
                and PlayerY < DoorY
                and screen[PlayerY + 1][PlayerX + 1] != "|"
                and screen[PlayerY + 1][PlayerX + 1] != "-"
                and screen[PlayerY + 1][PlayerX + 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY + 1][PlayerX + 1].isupper():
                    RB.send_command("n")
                    continue
                RB.send_command("n")
                PlayerX += 1
                PlayerY += 1
            elif (
                PlayerX > DoorX
                and PlayerY < DoorY
                and screen[PlayerY + 1][PlayerX - 1] != "|"
                and screen[PlayerY + 1][PlayerX - 1] != "-"
                and screen[PlayerY + 1][PlayerX - 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY + 1][PlayerX - 1].isupper():
                    RB.send_command("b")
                    continue
                RB.send_command("b")
                PlayerX -= 1
                PlayerY += 1
            elif (
                PlayerX < DoorX
                and PlayerY > DoorY
                and screen[PlayerY - 1][PlayerX + 1] != "|"
                and screen[PlayerY - 1][PlayerX + 1] != "-"
                and screen[PlayerY - 1][PlayerX + 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY - 1][PlayerX + 1].isupper():
                    RB.send_command("u")
                    continue
                RB.send_command("u")
                PlayerX += 1
                PlayerY -= 1
            elif (
                PlayerX > DoorX
                and PlayerY > DoorY
                and screen[PlayerY - 1][PlayerX - 1] != "|"
                and screen[PlayerY - 1][PlayerX - 1] != "-"
                and screen[PlayerY - 1][PlayerX - 1] != "+"
                and FrameInfo.get_tile_below_player() != "+"
            ):
                if screen[PlayerY - 1][PlayerX - 1].isupper():
                    RB.send_command("y")
                    continue
                RB.send_command("y")
                PlayerX -= 1
                PlayerY -= 1
            elif (
                PlayerX < DoorX
                and screen[PlayerY][PlayerX + 1] != "|"
                and screen[PlayerY][PlayerX + 1] != "-"
            ):
                if screen[PlayerY][PlayerX + 1].isupper():
                    RB.send_command("l")
                    continue
                RB.send_command("l")
                PlayerX += 1
            elif (
                PlayerX > DoorX
                and screen[PlayerY][PlayerX - 1] != "|"
                and screen[PlayerY][PlayerX - 1] != "-"
            ):
                if screen[PlayerY][PlayerX - 1].isupper():
                    RB.send_command("h")
                    continue
                RB.send_command("h")
                PlayerX -= 1
            elif (
                PlayerY < DoorY
                and screen[PlayerY + 1][PlayerX] != "-"
                and screen[PlayerY + 1][PlayerX] != "|"
            ):
                if screen[PlayerY + 1][PlayerX].isupper():
                    RB.send_command("j")
                    continue
                RB.send_command("j")
                PlayerY += 1
            elif (
                PlayerY > DoorY
                and screen[PlayerY - 1][PlayerX] != "-"
                and screen[PlayerY - 1][PlayerX] != "|"
            ):
                if screen[PlayerY - 1][PlayerX].isupper():
                    RB.send_command("k")
                    continue
                RB.send_command("k")
                PlayerY -= 1
            if RB.game_over() == True:
                break
            if stackcheck >= 100 and stack == False:
                stack = True
                GlobalStackCheck = True
                # print("Warning, Maybe Stack while going to door.")
                # debug_print(nowRoomID,goDoorid)
                break
        else:
            break
    if RB.game_over() is True:
        return -1, -1
    while wrongscreen():
        screen_refresh()

    return PlayerY, PlayerX


def debug_print(nowRoomID, goDoorid):
    print("**********---------------------***********")
    screen = RB.get_screen()
    Screenprint(screen)
    FrameInfo = RBParser.parse_screen(screen)
    PlayerY, PlayerX = FrameInfo.get_player_pos()
    PlayerY += 1
    print("DoorList")
    print("NowRoomID: {0}".format(nowRoomID))
    for ridx in range(len(RoomInfoList)):
        if VisitedRoom[ridx]:
            print("RoomID: {0}".format(ridx))
            print(
                "LeftUp Y {0}, X{1},".format(
                    RoomInfoList[ridx].leftup[0], RoomInfoList[ridx].leftup[1]
                )
            )
            print(
                "RightBottom Y {0}, X{1},".format(
                    RoomInfoList[ridx].rightbottom[0], RoomInfoList[ridx].rightbottom[1]
                )
            )
            print("Stairs Existed{0}".format(RoomInfoList[ridx].stairexisted))
            i = 0
            for i in range(len(RoomInfoList[ridx].doorlist)):
                DoorID = RoomInfoList[ridx].doorlist[i].id
                DoorX = RoomInfoList[ridx].doorlist[i].X
                DoorY = RoomInfoList[ridx].doorlist[i].Y
                CanThrough = (
                    len(RoomInfoList[ridx].doorlist[i].passagelist) != 0
                    and RoomInfoList[ridx].doorlist[i].visited
                )
                print(DoorID, DoorY, DoorX, CanThrough)
            print("Stairs Room distance: {0}".format(RoomInfoList[ridx].stairsRoomdis))
    print(
        "Now PlayerY,PlayerX : ({0},{1}) Aiming DoorID {2}".format(
            PlayerY, PlayerX, goDoorid
        )
    )
    for i in range(len(RoomInfoList[nowRoomID].itemList)):
        itemy, itemx, obj = RoomInfoList[nowRoomID].itemList[i]
        print(
            "Item ID:{0}, ItemY:{1}, ItemX:{2}, ItemChar:{3}".format(
                id, itemy, itemx, obj
            )
        )
    print(
        "Reverse Passage : GoDoorID:{0}, Passage:{1}".format(
            RoomInfoList[nowRoomID].returnPassage[0],
            RoomInfoList[nowRoomID].returnPassage[1],
        )
    )

    print("**********---------------------***********")


def screen_refresh():
    global FrameInfo
    RB.send_command("i")
    screen = RB.get_screen()
    # Screenprint(screen)
    FrameInfo = RBParser.parse_screen(screen)


# 通ったことのある通路の情報に従って別の部屋に移動する。
def move(passage):
    global RoomInfoList
    global FrameInfo
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    i = 0
    while i < len(passage):
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if RB.game_over():
            return
        while wrongscreen():
            screen_refresh()
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.05)
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        if screen[PlayerY + dy[passage[i]]][PlayerX + dx[passage[i]]].isupper():
            CloseCombatinpassage()
            # RB.send_command(command[passage[i]])
            continue
        RB.send_command(command[passage[i]])
        i += 1
    RB.send_command(command[passage[-1]])


# 今の部屋から行ける扉について,行ったことのない扉を探索する。
# 即ち,NNが新しい場所へ探索すると決めたら、この関数を実行してもらう。
def explore(nowRoomID, GoDoorID, Beforepassagecount, playery, playerx):
    global RoomInfoList
    global FrameInfo
    global GlobalStackCheck
    global VisitedRoom
    goDoorid = GoDoorID
    Passage = []
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    RoomInfoList[nowRoomID].doorlist[goDoorid].visited = True
    # 扉までついたら,右手法に従って通路を進んでもらう.
    # 通路を実際に進むところは丸ごとRightMethod内で行なっている
    Passage, ArrivalDoory, ArrivalDoorx, Prevdir = RightMethod(playery, playerx)
    if ExploreStack is True:
        print("Explore Stack...")
        sleep(3)
        GlobalStackCheck = True
        return [-1]
    # 右手法が終わった後,辿ってきた経路について...
    # ①空の場合,そもそも別の部屋に行けないので,この部屋のこの扉そのものをなかった事にする。
    if Passage == []:
        RoomInfoList[nowRoomID].doorlist[goDoorid].visited = True
        RoomInfoList[nowRoomID].doorlist[goDoorid].useless = True
    # ②空では無い場合、まず辿り着いた部屋は行った事があるかどうかを確認する関数checkVisitedを使ってたどり着いた部屋のIDを取得する。
    # 部屋のIDは特に決まった順序があるわけではなく,見つけた順に若いIDが割り当てられる.
    else:
        screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if RB.game_over():
            return [-1]
        while wrongscreen():
            screen_refresh()
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.05)
        PlayerY, PlayerX = FrameInfo.get_player_pos()
        PlayerY += 1
        PlayerY += dy[Prevdir]
        PlayerX += dx[Prevdir]
        goRoomid = checkVisited(PlayerY, PlayerX)
        if goRoomid == nowRoomID:
            RoomInfoList[nowRoomID].doorlist[goDoorid].useless = True
            Passage = []
            return Passage
        Afterpassagecount = FrameInfo.get_tile_count("#")
        reversePassage = Passage.copy()
        reversePassage.reverse()
        for r in range(len(reversePassage)):
            reversePassage[r] = (reversePassage[r] + 2) % 4
        getDoorid = -1
        RoomInfoMake(PlayerY, PlayerX, screen, goRoomid, (getDoorid, reversePassage))
        CloseCombatinroom(goRoomid)
        RB.send_command(command[Prevdir])
        for d in range(len(RoomInfoList[goRoomid].doorlist)):
            doorY = RoomInfoList[goRoomid].doorlist[d].Y
            doorX = RoomInfoList[goRoomid].doorlist[d].X
            # print("({0},{1}), ({2},{3}),".format(ArrivalDoory,ArrivalDoorx,doorY,doorX))
            if doorY == ArrivalDoory and doorX == ArrivalDoorx:
                getDoorid = d
        # if(RoomInfoList[goRoomid].returnPassage[0]==-1):
        #    VisitedRoom[goRoomid]=False
        RoomInfoMake(PlayerY, PlayerX, screen, goRoomid, (getDoorid, reversePassage))
        didx = 0
        if Afterpassagecount - Beforepassagecount == len(Passage) - 1:
            # print("One Road!")
            for door in RoomInfoList[goRoomid].doorlist:
                # if(door.Y == PlayerY+dy[(Passage[-1]+2)%4] and door.X == PlayerX+dx[(Passage[-1]+2)%4]):
                if door.Y == ArrivalDoory and door.X == ArrivalDoorx:
                    reversePassage = Passage.copy()
                    reversePassage.reverse()
                    for r in range(len(reversePassage)):
                        reversePassage[r] = (reversePassage[r] + 2) % 4
                    door.visited = True
                    door.passagelist.append(
                        PassageInfo(
                            len(RoomInfoList[goRoomid].doorlist[didx].passagelist),
                            reversePassage,
                            nowRoomID,
                        )
                    )
                didx += 1
        RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist.append(
            PassageInfo(
                len(RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist),
                Passage,
                goRoomid,
            )
        )
    screen = RB.get_screen()
    # debug_print(goRoomid,0)
    return Passage


# 探索後に辿り着いた部屋のIDを調べる。
# 行ったことのある部屋ならその部屋のIDを返し,行った事が無い部屋なら,現在の部屋ID+1を設定する。
def checkVisited(playerY, playerX):
    global maxRoomID
    for i in range(15):
        if (
            playerY >= RoomInfoList[i].leftup[0]
            and playerY <= RoomInfoList[i].rightbottom[0]
        ) and (
            playerX >= RoomInfoList[i].leftup[1]
            and playerX <= RoomInfoList[i].rightbottom[1]
        ):
            return RoomInfoList[i].id
    maxRoomID += 1
    return maxRoomID


def eval_network(net, net_input):
    assert len(net_input) == 10
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
    # print(total_reward / run_neat_base.n)
    # print("<-- Episode finished after average {} time-steps with reward {}".format(t + 1//5, total_reward / run_neat_base.n))
    # print("Fin")


def demoplay_learned_genome(genome_config):
    global demoplay_mode
    demoplay_mode = True
    method = "RNSGA2"
    datanum = 30
    trynum = 5
    path = "./"
    method_list = []
    method_list.append(method)
    dirs = []
    network_dirs = []
    generation = 150
    all_files = os.listdir(path)
    all_dirs = [f for f in all_files if os.path.isdir(os.path.join(path, f))]
    for i in range(len(method_list)):
        method_com = method_list[i]
        files = os.listdir(path)
        dirs.append([f for f in all_dirs if f.startswith(method_com)])
        network_dirs.append([])
        # print(dirs)
        for dirpath in dirs[i]:
            files = os.listdir(dirpath)
            dirs_indir = [f for f in files if os.path.isdir(os.path.join(dirpath, f))]
            # print(files_indir)
            gen_data_dir = [f for f in dirs_indir if f == "Gen_" + str(generation)]
            if len(gen_data_dir) == 1:
                network_dirs[i].append(dirpath + "/" + gen_data_dir[0] + "/Networks")
                if len(network_dirs[i]) == datanum:
                    break
    all_networks = []
    elite_networks = []
    battle_field = []
    idx = 0
    for battle in range(datanum):
        for method in range(len(method_list)):
            one_network_dir = network_dirs[method][battle]
            networks = os.listdir(one_network_dir)
            print(one_network_dir)
            for i in range(len(networks)):
                with open(one_network_dir + "/" + networks[i], "rb") as f:
                    c1 = pickle.load(f)
                    all_networks.append(c1)
                    battle_field.append((idx, c1.fitness))
                idx += 1
    first_front, paretofront_indices = sortNondominated2(
        battle_field, len(battle_field), False
    )
    # print(len(networks1))
    # print(len(networks2))
    # print(first_front)
    # print(first_idx)
    print(paretofront_indices[0])
    for rank in range(len(paretofront_indices)):
        # elite_networks.append([])
        for idx in paretofront_indices[rank]:
            # elite_networks[-1].append(all_networks[idx])
            elite_networks.append(all_networks[idx])
    print(elite_networks)
    # elite_networks_fitnesses = [[]]
    elite_networks_fitnesses = [[]]
    """
    for rank in range(len(elite_networks)):
        # elite_networks_fitnesses.append([])
        for net in elite_networks[rank]:
            # elite_networks_fitnesses[-1].append(net.fitness)
            elite_networks_fitnesses.append(net.fitness)
    """
    for net in elite_networks:
        elite_networks_fitnesses[-1].append(net.fitness)
    print(elite_networks_fitnesses)
    neat.visualize.plot_stats3D(
        150,
        elite_networks_fitnesses,
        filename="RNSGA2_elite_networks_fitness",
        title="30 trials 150 Gen all network fitnesses",
    )
    ref_points = [[1.0, 1.0, 1.0], [1.0, 1.0, 0], [0, 1.0, 1.0]]
    np_ref_points = np.array(ref_points, dtype=float)
    elite_networks_np_fitnesses = np.array(
        [fitness for fitness in elite_networks_fitnesses[0]]
    )
    print(elite_networks_np_fitnesses.shape)
    dist_to_ref_points = calc_norm_pref_distance(
        elite_networks_np_fitnesses, np_ref_points
    )
    dist_to_ref_points_rank = np.argmin(dist_to_ref_points, axis=1)
    print(dist_to_ref_points_rank)
    ref_points_cluster_fitnesses = [[] for i in range(len(ref_points))]
    ref_points_cluster_networks = [[] for i in range(len(ref_points))]
    print(ref_points_cluster_fitnesses)
    for idx in range(len(elite_networks_fitnesses[0])):
        ref_points_cluster_fitnesses[dist_to_ref_points_rank[idx]].append(
            elite_networks_fitnesses[0][idx]
        )
        ref_points_cluster_networks[dist_to_ref_points_rank[idx]].append(
            (idx, elite_networks[idx])
        )
    balance_networks, balance_indices = sortNondominated2(
        ref_points_cluster_networks[0], 15
    )
    stairs_networks, stairs_indices = sortNondominated2(
        ref_points_cluster_networks[1], 15
    )
    item_networks, item_indices = sortNondominated2(ref_points_cluster_networks[2], 15)
    print(balance_networks)
    for front in balance_networks:
        assignCrowdingDist(front)
    balance_chosen = list(chain(*balance_networks[:-1]))
    k = 15 - len(balance_chosen)
    if k > 0:
        # sorted_front = sorted(pareto_fronts[-1], attrgetter("crowding_dist"), reverse=True)
        sorted_front = sorted(
            balance_networks[-1], key=lambda x: x[1].crowding_dist, reverse=True
        )
        balance_chosen.extend(sorted_front[:k])
    for front in stairs_networks:
        assignCrowdingDist(front)
    stairs_chosen = list(chain(*stairs_networks[:-1]))
    k = 15 - len(stairs_chosen)
    if k > 0:
        # sorted_front = sorted(pareto_fronts[-1], attrgetter("crowding_dist"), reverse=True)
        sorted_front = sorted(
            stairs_networks[-1], key=lambda x: x[1].crowding_dist, reverse=True
        )
        stairs_chosen.extend(sorted_front[:k])
    for front in item_networks:
        assignCrowdingDist(front)
    item_chosen = list(chain(*item_networks[:-1]))
    k = 15 - len(item_chosen)
    if k > 0:
        # sorted_front = sorted(pareto_fronts[-1], attrgetter("crowding_dist"), reverse=True)
        sorted_front = sorted(
            item_networks[-1], key=lambda x: x[1].crowding_dist, reverse=True
        )
        item_chosen.extend(sorted_front[:k])
    print("item_chosen")
    print(item_chosen)
    print("balance_chosen")
    print(balance_chosen)
    top90_3strategy_networks_fitnesses = [[] for i in range(len(ref_points))]
    for idx, _ in balance_chosen:
        top90_3strategy_networks_fitnesses[0].append(
            # elite_networks_fitnesses[0][ref_points_cluster_networks[0][idx][0]]
            elite_networks_fitnesses[0][idx]
        )
    for idx, _ in stairs_chosen:
        top90_3strategy_networks_fitnesses[1].append(
            # elite_networks_fitnesses[0][ref_points_cluster_networks[1][idx][0]]
            elite_networks_fitnesses[0][idx]
        )
    for idx, _ in item_chosen:
        top90_3strategy_networks_fitnesses[2].append(
            # elite_networks_fitnesses[0][ref_points_cluster_networks[2][idx][0]]
            elite_networks_fitnesses[0][idx]
        )
    neat.visualize.plot_stats3D(
        150,
        ref_points_cluster_fitnesses,
        filename="3strategy_networks_fitness",
        title="3 strategy networks fitnesses",
    )
    neat.visualize.plot_stats3D(
        150,
        top90_3strategy_networks_fitnesses,
        filename="top45_3strategy_networks_fitness",
        title="Top 45 3 strategy networks fitnesses",
    )

    balance_networks = get_reference_neighobor_points(
        elite_networks, [1.0, 1.0, 1.0], 15
    )
    noitem_networks = get_reference_neighobor_points(elite_networks, [1.0, 1.0, 0], 15)
    nostairs_networks = get_reference_neighobor_points(
        elite_networks, [0, 1.0, 1.0], 15
    )
    three_strategy_networks = []
    three_strategy_networks.append([])
    for net in balance_networks:
        three_strategy_networks[-1].append(net.fitness)
    three_strategy_networks.append([])
    for net in noitem_networks:
        three_strategy_networks[-1].append(net.fitness)
    three_strategy_networks.append([])
    for net in nostairs_networks:
        three_strategy_networks[-1].append(net.fitness)
    print(three_strategy_networks)
    # neat.visualize.plot_stats3D(
    #    150,
    #    three_strategy_networks,
    #    filename="3_strategy_networks_fitness",
    #    title="3 strategy networks fitnesses",
    # )
    max_down = 0
    min_room = 100
    min_item = 100
    max_item = 0
    usenet_idx = 0
    for idx in range(len(elite_networks)):
        # for idx in range(len(all_networks)):
        # print(elite_networks[idx].fitness)
        # if elite_networks[idx].fitness[1] > max_room:
        if (
            elite_networks[idx].fitness[0] > max_down
            and elite_networks[idx].fitness[2] > max_item
        ):
            # max_room = elite_networks[idx].fitness[1]
            max_down = elite_networks[idx].fitness[0]
            # min_room = elite_networks[idx].fitness[1]
            max_item = elite_networks[idx].fitness[2]
            usenet_idx = idx
    print("Now use network fitness: {}".format(elite_networks[usenet_idx].fitness))
    net = neat.nn.FeedForwardNetwork.create(elite_networks[usenet_idx], genome_config)
    for _ in range(trynum):
        RogueTrying(net, True)


def calc_norm_pref_distance(A, B, weights=None):
    if weights is None:
        weights = np.full(len(A[0]), 1 / len(A[0]))
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    N = (D ** 2) * weights
    N = np.sqrt(np.sum(N, axis=1) * len(weights))
    return np.reshape(N, (A.shape[0], B.shape[0]))


def assignCrowdingDist(individuals):

    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind[1].fitness, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0][1].fitness)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i][1].crowding_dist = dist


def get_reference_neighobor_points(individuals, ref_point, outnum):
    indsfitness = []
    ret_networks = []
    for i, net in enumerate(individuals):
        ref_dist = 0.0
        for d in range(len(ref_point)):
            ref_dist += (net.fitness[d] - ref_point[d]) ** 2
        ref_dist = math.sqrt(ref_dist)
        indsfitness.append((i, ref_dist))
    indsfitness.sort(key=lambda x: x[1])
    print(indsfitness)
    for i in range(outnum):
        ret_networks.append(individuals[indsfitness[i][0]])
    return ret_networks


def RogueTrying(net, demo=False):
    fitness = [0.0, 0.0, 0.0]
    priority_fitness = 0.0
    global RightActnum
    global ExploreActnum
    global PrevAct
    global FrameInfo
    global now_inventory
    global demoplay_mode
    demoplay_mode = demo
    for _ in range(run_neat_base.n):
        Roguetime = 0
        Initialize()
        # print("--> Starting new episode")
        t = 0
        Roguetime += 1
        # print(Roguetime)
        random.seed()
        # rn=random.random()*1000000
        rn = random.random() * 1000
        rn = math.floor(rn)
        run_neat_base.env.seed(rn)
        run_neat_base.env.reset()
        # observation = run_neat_base.env.reset()
        # inputs =observation
        # inputs = inputs.flatten()
        # inputs = inputs.reshape(inputs.size,1)
        nowRoomID = 0
        screen_refresh()
        screen = RB.get_screen()
        FrameInfo = RBParser.parse_screen(screen)
        if demoplay_mode:
            Screenprint(screen)
            sleep(0.005)
        # action = eval_network(net, inputs)
        if RB.game_over() is True:
            # print("Game Over.")
            fitness = [
                x + y
                for (x, y) in zip(fitness, [-1.0, maxRoomID / 5.0, Getitemnum / 6.0])
            ]
            break
        while wrongscreen():
            screen_refresh()
        screen = RB.get_screen()
        Playery, Playerx = FrameInfo.get_player_pos()
        Playery += 1
        done = False
        while not done:
            # run_neat_base.env.render()
            t += 1
            # RogueBoxオブジェクトを呼び出すときは,env.unwrapped.rb
            screen = RB.get_screen()
            if RB.game_over() == True:
                # print("Game Over.")
                fitness = [
                    x + y
                    for (x, y) in zip(
                        fitness, [-1.0, maxRoomID / 5.0, Getitemnum / 6.0]
                    )
                ]
                break
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            Playery, Playerx = FrameInfo.get_player_pos()
            Playery += 1
            nowRoomID = checkVisited(Playery, Playerx)
            RoomInfoMake(Playery, Playerx, screen, nowRoomID, (-1, []))
            if demoplay_mode:
                Screenprint(screen)
                sleep(0.005)
            CloseCombatinroom(nowRoomID)
            # debug_print(nowRoomID,0)
            if StairsFound == False:
                StairsCheck(nowRoomID)
            Prevdis = RoomInfoList[nowRoomID].stairsRoomdis
            Input = MakeInput(nowRoomID, t)
            action = eval_network(net, Input)
            PrevAct = action
            # RoominfoPrint(nowRoomID)
            """
            print("----------Before Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """

            Safe, Stairdown = ActMethod(action, nowRoomID)
            if GlobalStackCheck == True:
                # print("Unfortunately Stacking. Retry.")
                return RogueTrying(net)
            # 次の部屋へいくための道を取得してもらう。
            # 尚、ここで求めた道が明確に次の部屋と一本道である場合、次の部屋の対応する扉について、
            # 求めた道を反転した道を次の部屋の既知の扉と道の情報とする。
            # 明示的に一本道であると確定するには,get_tile_count('#')で探索前後における#の数を数える。
            if RB.game_over() == True:
                # print("Game Over.")
                fitness = [
                    x + y
                    for (x, y) in zip(
                        fitness, [0.0, min(1.0, maxRoomID / 5.0), Getitemnum / 5.0]
                    )
                ]
                break
            while wrongscreen():
                screen_refresh()
            screen = RB.get_screen()
            FrameInfo = RBParser.parse_screen(screen)
            Playery, Playerx = FrameInfo.get_player_pos()
            Playery += 1
            if demoplay_mode:
                Screenprint(screen)
                sleep(0.005)
            if Safe is False:
                if demoplay_mode:
                    print("Wrong Act...")
                    Screenprint(screen)
                    sleep(0.005)
                    sleep(1.0)
                done = True
            else:
                RightActnum += 1
                if action == 1:
                    ExploreActnum += 1
            if Safe == True and Stairdown == True:
                # fitness = [ x + y for (x, y) in zip(fitness,[(20.0-ExploreActnum)/20.0,ExtraExplore/10.0,Getitemnum/6.0]) ]
                fitness = [
                    x + y
                    for (x, y) in zip(
                        fitness,
                        [
                            (20.0 - ExtraExplore) / 20.0,
                            min(1.0, maxRoomID / 5.0),
                            Getitemnum / 5.0,
                        ],
                    )
                ]
                priority_fitness += 1.0
                # print("Get down stairs! Add Fitness: {0}".format([(20.0-ExploreActnum)/20.0,ExtraExplore/10.0,Getitemnum/6.0]))
                Initialize()
                t = 0
                done = True
                break

            if "Hungry" in screen and FoodNum > 0:
                IntelisenseItemuse(-1, 0, "Eat")
            nowRoomID = checkVisited(Playery, Playerx)
            if StairsFound == True and RoomInfoList[nowRoomID].stairsRoomdis == 100:
                StairsRoomdisCheck(nowRoomID, Prevdis)

            """
            print("----------After Action----------")
            RoominfoPrint(nowRoomID)
            Screenprint(screen)
            """
            # sleep(0)
            if done or t >= 20:
                if Safe == True and t < 20:
                    fitness = [
                        x + y
                        for (x, y) in zip(
                            fitness, [ExploreActnum, ExtraExplore, RightActnum]
                        )
                    ]
                    print("???")

                    # fitness = [30-ExtraExplore,ExtraExplore,RightActnum]
                if t == 20 or Safe == False:
                    fitness = [
                        x + y
                        for (x, y) in zip(
                            fitness, [0.0, min(1.0, maxRoomID / 5.0), Getitemnum / 5.0]
                        )
                    ]
                    # fitness = [ x + y for (x, y) in zip(fitness,[0.0,Defeatmonsnum/10.0,Getitemnum/6.0]) ]
                    # fitness = [-50,-50,RightActnum]
                # print(fitness)
                break
    # print(total_reward / run_neat_base.n)
    # print("<-- Episode finished after average {} time-steps with reward {}".format(t + 1//5, total_reward / run_neat_base.n))
    # print("Fin")
    fitness = [
        x / y
        for (x, y) in zip(fitness, [run_neat_base.n, run_neat_base.n, run_neat_base.n])
    ]
    priority_fitness /= run_neat_base.n
    return fitness, priority_fitness


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
    global Getitemnum
    global Defeatmonsnum
    global Moveonlynum
    global now_inventory
    global stepcnt
    ExploreStack = False
    GlobalStackCheck = False
    Getitemnum = 0
    Defeatmonsnum = 0
    ExtraExplore = 0
    Moveonlynum = 0
    RightActnum = 0
    ExploreActnum = 0
    StairsFound = False
    maxRoomID = -1
    now_inventory = None
    RB = run_neat_base.env.unwrapped.rb
    RBParser = RB.parser
    RBParser.reset()
    RoomInfoList.clear()
    stepcnt = 0
    VisitedRoom = np.array([False] * 15)
    for k in range(15):
        RoomInfoList.append(
            RoomInfo(k, (-1, -1), (-1, -1), [], [], False, 100, (-1, []))
        )


def learn(env, config_path, demo_play):
    run_neat_base.run(
        eval_network,
        eval_single_genome,
        demoplay_learned_genome,
        environment=env,
        config_path=config_path,
        demo_play=demo_play,
    )


def GotoStairsAct(nowRoomID, screen):
    if RoomInfoList[nowRoomID].stairexisted == True:
        while wrongscreen():
            screen_refresh()
        StairsCoord = FrameInfo.get_list_of_positions_by_tile("%")
        (StairsY, StairsX) = StairsCoord[0]
        GotoObject(nowRoomID, StairsY, StairsX, "%")
        if RB.game_over():
            return False
        return True
    else:
        return False


def GotoKnownRoomAct(nowRoomID, GoDoorID):
    # print(len(RoomInfoList[nowRoomID].doorlist))
    if GoDoorID + 1 > len(RoomInfoList[nowRoomID].doorlist):
        # print("GoDoorID is too large!")
        return False
    elif not RoomInfoList[nowRoomID].doorlist[GoDoorID].visited:
        # print("Door ID:GoDoorID didn't visit!")
        return False
    elif RoomInfoList[nowRoomID].doorlist[GoDoorID].useless:
        return False
    GoDoorX = RoomInfoList[nowRoomID].doorlist[GoDoorID].X
    GoDoorY = RoomInfoList[nowRoomID].doorlist[GoDoorID].Y
    GoPassage = RoomInfoList[nowRoomID].doorlist[GoDoorID].passagelist[0].passage
    GotoDoor(nowRoomID, GoDoorID, GoDoorY, GoDoorX)
    if RB.game_over() == True:
        return False
    move(GoPassage)
    return True


def GotoPrevRoom(nowRoomID, reversePassage):
    global Moveonlynum
    GoDoorID = reversePassage[0]
    GoPassage = reversePassage[1]
    if reversePassage[0] == -1 or reversePassage[1] == [] or Moveonlynum >= 4:
        # print("There is no reverse passage! (only RoomID:0)")
        return False
    else:
        Moveonlynum += 1
        GoDoorX = RoomInfoList[nowRoomID].doorlist[GoDoorID].X
        GoDoorY = RoomInfoList[nowRoomID].doorlist[GoDoorID].Y
        GotoDoor(nowRoomID, GoDoorID, GoDoorY, GoDoorX)
    if RB.game_over() == True:
        return False
    move(GoPassage)
    return True


def GotoStairsRoomAct(nowRoomID):
    if StairsFound == False or RoomInfoList[nowRoomID].stairexisted == True:
        return False
    Mindis = 100
    GoDoorID = -1
    for d in range(len(RoomInfoList[nowRoomID].doorlist)):
        if (
            RoomInfoList[nowRoomID].doorlist[GoDoorID].visited
            and len(RoomInfoList[nowRoomID].doorlist[d].passagelist) > 0
            and RoomInfoList[
                RoomInfoList[nowRoomID].doorlist[d].passagelist[0].connectroomid
            ].stairsRoomdis
            < Mindis
        ):
            Mindis = RoomInfoList[
                RoomInfoList[nowRoomID].doorlist[d].passagelist[0].connectroomid
            ].stairsRoomdis
            GoDoorID = d
    if GoDoorID == -1 or RB.game_over():
        return False
    return GotoKnownRoomAct(nowRoomID, GoDoorID)


def FightAct(screen, nowRoomID):
    global Defeatmonsnum
    Enemynum = RoomObjectSearch(
        screen,
        RoomInfoList[nowRoomID].leftup,
        RoomInfoList[nowRoomID].rightbottom,
        "monster",
    )
    if Enemynum == 0:
        return False
    for _ in range(50):
        Playery = RB.player_pos[0] + 1
        Playerx = RB.player_pos[1]
        screen = RB.get_screen()
        for dir in range(4):
            if screen[Playery + dy[dir]][Playerx + dx[dir]].isupper():
                RB.send_command(command[dir])
                screen = RB.get_screen()
                if "defeat" in screen[0]:
                    Defeatmonsnum += 1
                # Screenprint(screen)
        Enemynum = RoomObjectSearch(
            screen,
            RoomInfoList[nowRoomID].leftup,
            RoomInfoList[nowRoomID].rightbottom,
            "monster",
        )
        if Enemynum == 0:
            break
    return True


def CloseFight(Direction):
    RB.send_command("f")
    RB.send_command(command[Direction])


def ExploreAct(nowRoomID):
    global Moveonlynum
    global FrameInfo
    while wrongscreen():
        screen_refresh()
    py, px = FrameInfo.get_player_pos()
    py += 1
    GoDoorID, GoDoorX, GoDoorY = NotvisitedDoorCheck(nowRoomID, py, px)
    while wrongscreen():
        screen_refresh()
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    Beforepassagecount = FrameInfo.get_tile_count("#")
    if GoDoorID != -1 and GoDoorX != -1 and GoDoorY != -1:
        # print("There is unvisited Door! Explore Start!")
        Moveonlynum = 0
        py, px = GotoDoor(nowRoomID, GoDoorID, GoDoorY, GoDoorX)
        if RB.game_over():
            return False
        Passage = explore(nowRoomID, GoDoorID, Beforepassagecount, py, px)
        if RB.game_over():
            # print("Game_Over...")
            # screen = RB.get_screen()
            # Screenprint(screen)
            # sleep(3)
            return False
        if len(Passage) > 0 and Passage[0] == -1:
            # print("Can't explore not visited door. End.")
            # sleep(3)
            return False
        else:
            return True
    else:
        """
        for d in range(len(RoomInfoList[nowRoomID].doorlist)):
            if len(RoomInfoList[nowRoomID].doorlist[d].passagelist) > 0:
                nxtroom = RoomInfoList[
                    RoomInfoList[nowRoomID].doorlist[d].passagelist[0].connectroomid
                ]
                for nd in range(len(nxtroom.doorlist)):
                    if not nxtroom.doorlist[nd].visited:
                        # print("there isn't not visited door, but Go to known room.")
                        # Screenprint(screen)
                        GotoKnownRoomAct(nowRoomID, d)
                        return True
        """
        return False


def wrongscreen():
    global FrameInfo
    screen = RB.get_screen()
    FrameInfo = RBParser.parse_screen(screen)
    if RB.game_over():
        return False
    return FrameInfo.get_player_pos() is None or RB.screen_trouble()


def PickupAct(nowRoomID):
    global FrameInfo
    while wrongscreen():
        screen_refresh()
    py, px = FrameInfo.get_player_pos()
    py += 1
    PickupSearch(nowRoomID, py, px)
    if not RoomInfoList[nowRoomID].itemList:
        return False
    itemy, itemx, obj = RoomInfoList[nowRoomID].itemList[0]
    GotoObject(nowRoomID, itemy, itemx, obj)
    if RB.game_over():
        return False
    return True


def ActMethod(action, nowRoomID):
    global ExtraExplore
    """
    if(action<4):
        if(StairsFound==True):
            ExtraExplore+=1
        #print("GotoKnownRoomAct ID: {0}!".format(action))
        return GotoKnownRoomAct(nowRoomID,action),False
    """
    if action == 0:
        # if(StairsFound==True):
        # ExtraExplore+=1
        # print("Goto Prev Room!")
        if demoplay_mode is True:
            sys.stdout.write("\rGoto Prev Room!\n")
            sys.stdout.flush()
            print("\033[1A", end="")
        return GotoPrevRoom(nowRoomID, RoomInfoList[nowRoomID].returnPassage), False
    elif action == 1:
        if StairsFound == True:
            ExtraExplore += 1
        if demoplay_mode is True:
            sys.stdout.write("\rExplore!\n")
            sys.stdout.flush()
            print("\033[1A", end="")
            sleep(1)
        return ExploreAct(nowRoomID), False
    elif action == 2:
        screen = RB.get_screen()
        # print("Fight!")
        if demoplay_mode is True:
            sys.stdout.write("\rFight!\n")
            sys.stdout.flush()
            print("\033[1A", end="")
            # sleep(1)
        return FightAct(screen, nowRoomID), False
    # elif(action==3):
    # screen=RB.get_screen()
    # print("GotoStairs!")
    # return GotoStairsAct(nowRoomID,screen),True
    elif action == 3:
        if RoomInfoList[nowRoomID].stairexisted == True:
            # print("GotoStairs!")
            if demoplay_mode is True:
                sys.stdout.write("\rGoto Stairs!\n")
                sys.stdout.flush()
                print("\033[1A", end="")
                sleep(1)
            screen = RB.get_screen()
            return GotoStairsAct(nowRoomID, screen), True
        else:
            # print("GotoStairsRoom!")
            if demoplay_mode is True:
                sys.stdout.write("\rGoto Stairs Room!\n")
                sys.stdout.flush()
                print("\033[1A", end="")
                sleep(1)
            return GotoStairsRoomAct(nowRoomID), False
    elif action == 4:
        # print("Pickup item!")
        if demoplay_mode is True:
            sys.stdout.write("\rPick up item!\n")
            sys.stdout.flush()
            print("\033[1A", end="")
            sleep(1)
        return PickupAct(nowRoomID), False


def isdominates(fitnesses1, fitnesses2):
    """Return true if each objective of *self* is not strictly worse than
    the corresponding objective of *other* and at least one objective is
    strictly better.

    :param obj: Slice indicating on which objectives the domination is
                tested. The default value is `slice(None)`, representing
                every objectives.
    """
    not_equal = False
    for self_fitnesses, other_fitnesses in zip(fitnesses1, fitnesses2):
        if self_fitnesses > other_fitnesses:
            not_equal = True
        elif self_fitnesses < other_fitnesses:
            return False
    return not_equal


def sortNondominated2(individuals, k, first_front_only=False, reverse=False):
    if k == 0:
        return []
    fits = []
    if type(individuals[0][1]) is list:
        for ind in individuals:
            fits.append(ind[1])
    else:
        for ind in individuals:
            fits.append(ind[1].fitness)

    print(len(fits))
    current_front = []
    next_front = []
    fronts = [[]]
    current_front_indices = []
    next_front_indices = []
    fronts_indices = [[]]
    dominating_indices = [[] for _ in range(len(individuals))]
    n_dominated = np.zeros(len(individuals))
    for i in range(len(fits)):
        # print("i, fit_i: {0}, {1}".format(i,fit_i))
        # print("map_fit_ind[fit_i]: {0}".format(map_fit_ind[fit_i]))
        for j in range(i + 1, len(fits)):
            if isdominates(fits[i], fits[j]):
                n_dominated[j] += 1
                dominating_indices[i].append(j)
            elif isdominates(fits[j], fits[i]):
                n_dominated[i] += 1
                dominating_indices[j].append(i)
        if n_dominated[i] == 0:
            current_front.append(fits[i])
            current_front_indices.append(i)

    # print(current_front_indices)

    for idx in current_front_indices:
        fronts_indices[-1].append(idx)
        fronts[-1].append(tuple(individuals[idx]))
    # print(fronts_indices)
    # print(fronts)
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            for i in current_front_indices:
                for j in dominating_indices[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front_indices.append(j)
                        next_front.append(tuple(individuals[j]))
                        pareto_sorted += 1
            fronts_indices.append(next_front_indices)
            fronts.append(next_front)
            current_front_indices = next_front_indices
            current_front = next_front
            next_front = []
            next_front_indices = []
    # print("END")
    return fronts, fronts_indices
