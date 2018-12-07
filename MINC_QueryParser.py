import sys
import os
import time, argparse
import ParseConfigFile as parseConfig
import QueryExecution as QExec
import re, gc

def setDelFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = True
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setUpdFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = True
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setInsFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = True
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setSelFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = True
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setJoinFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = True

def setLimFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = True
    joinFlag = False

def setOrderFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = True
    limFlag = False
    joinFlag = False

def setTabFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = True
    orderFlag = False
    limFlag = False
    joinFlag = False

def setHavFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = True
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setGrpFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = True
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setWhereFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = True
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setProjFlagTrue(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = True
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def setAllFlagsFalse(selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag):
    projFlag = False
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False

def checkFlagAndSetList(selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList, selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag, token):
    if projFlag:
        if len(projectList)==0:
            projectList.append(token)
        else:
            projectList.append(" " + token)
    elif whereFlag:
        if len(whereList)==0:
            whereList.append(token)
        else:
            whereList.append(" " + token)
    elif grpFlag:
        if len(groupByList)==0:
            groupByList.append(token)
        else:
            groupByList.append(" " + token)
    elif havFlag:
        if len(havingList)==0:
            havingList.append(token)
        else:
            havingList.append(" " + token)
    elif tabFlag:
        if len(tableList)==0:
            tableList.append(token)
        else:
            tableList.append(" " + token)
    elif orderFlag:
        if len(orderByList)==0:
            orderByList.append(token)
        else:
            orderByList.append(" " + token)
    elif limFlag:
        if len(limitList)==0:
            limitList.append(token)
        else:
            limitList.append(" " + token)
    elif joinFlag:
        if len(joinList)==0:
            joinList.append(token)
        else:
            joinList.append(" " + token)
    return whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList


def parseQueryOpWise(
        sessQuery, selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList,
        selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag, tableDict, tabColDict, tabColTypeDict,
        joinPairDict, joinPredBitPosDict, configDict):
    tokens = sessQuery.split()
    for token in tokens:
        token = token.lower()
        if "select" in token:
            se
    return (selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList)

def parseMINCQuery(tableDict, tabColDict, tabColTypeDict, joinPairDict, joinPredBitPosDict, sessQuery, configDict):
    # we are level-agnostic here because MINC has ad-hoc parentheses and query format
    selInsUpdDelList = []
    whereList = []
    projectList = []
    groupByList = []
    havingList = []
    tableList = []
    orderByList = []
    limitList = []
    joinList = []
    selFlag = False
    insFlag = False
    updFlag = False
    delFlag = False
    whereFlag = False
    projFlag = False
    grpFlag = False
    havFlag = False
    tabFlag = False
    orderFlag = False
    limFlag = False
    joinFlag = False
    (selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList) = parseQueryOpWise(
        sessQuery, selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList,
        selFlag, insFlag, updFlag, delFlag, whereFlag, projFlag, grpFlag, havFlag, tabFlag, orderFlag, limFlag, joinFlag, tableDict, tabColDict, tabColTypeDict,
        joinPairDict, joinPredBitPosDict, configDict)
    return (sessQuery, selInsUpdDelList, whereList, projectList, groupByList, havingList, tableList, orderByList, limitList, joinList)