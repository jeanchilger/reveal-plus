#!/usr/bin/env python
# -*- coding: UTF-8  -*-

"""docstring
"""

__revision__ = '0.1'

import sys,os
import getopt


# python doJudgementMain.py --topic=tr0 --judgefile=../judgement/qrels.oldreut.list --input=test.list --output=rel.1.Judged.doc.list
def usage():
    print """python yourFile.py
    --help
    --topicid=the topic id
    --judgefile=the relevance judgefile (qrels.robust04.txt)
    --input=the file to be judged
    --output=result of input
    """

def error():
    usage()
    sys.exit(-1)

def cmdProcess(argv):
    myArgs={
        "defaulArgument1":"",
    }
    try:
        opts, args = getopt.getopt(argv,"h",["help","input=","output=","topic=","judgefile=",'record='])
    except getopt.GetoptError:
        error()
    for opt, arg in opts:
        if opt in ("--help","-h"):
            usage()
            sys.exit()
        else:
            opt="".join(opt[2:])
            myArgs[opt]=arg
    return myArgs




def readJudgment(judgefile,topic):
    """
    Adiciona os documentos referentes ao tópico atual em um dicionário.
    A chave do dicionário é o número do documento e o valor é 1 se o documento
    é relevante e 0 caso contrário.
    """
    for line in open(judgefile):
        line=line.strip()
        if len(line)==0:
            continue
        info=line.split()
        assert len(info)==4
        (topicid,dummy,docno,res)=info
       # print("topicid,dummy,docno,res topic", topicid,dummy,docno,res, "topic ", topic)
        # judge=judgement.setdefault(topicid,{})
        # assert docno not in judge
        if topicid == topic:
            judge.setdefault(docno,res)
            #print("topicid,dummy,docno,res topic", topicid,dummy,docno,res, "topic ", topic)

def doJudge(docno, topicid):

    """
    Retorna se o documento atual é positivo (1) ou negativo (0).
    Retorna -1 se o documento não foi encontrado.
    """
    if docno in judge:
        return judge[docno]
    else:
        return -1

if __name__=="__main__":

    argvNum=1
    if len(sys.argv)<=argvNum:
        error()
    myArgs=cmdProcess(sys.argv[1:])

    judge={}

    topicid=myArgs['topic']
    inf=myArgs["input"]
    outf=myArgs['output']
    recordf=myArgs['record']
    out=open(outf,"w")
    record=open(recordf,"a")

    readJudgment(myArgs['judgefile'],topicid)
    for line in open(inf):
        line=line.strip()
        if len(line)==0:
            continue
        docno=line
        #do judged one by one
        #callJudgeApi(docno, topicid, memoryf, myArgs['judgefile'])

        res=doJudge(docno, topicid)
        if int(res)>0:
            #out.write("%s %s\n"%(docno,res))

            out.write("%s\n"%(docno))
            record.write("%s 1\n"%(docno))
        else:
            record.write("%s 0\n"%(docno))
    out.close()
    record.close()
