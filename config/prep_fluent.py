import os
import sys
import csv
from collections import Counter

csvdir = '/users/spraak/spchdata/FluentSpeechCommands/data'
outdir = '/esat/spchdisk/scratch/hvanhamm/FluentSpeechCommands/assist' # temp storage
finaldir = '/users/spraak/spchdata/FluentSpeechCommands/' # where data will be found by users

cvt={'change language,none,none':'<changelang2/>',
     'change language,German,none':'<changelang language="German"/>',
     'change language,Korean,none':'<changelang language="Korean"/>',
     'change language,Chinese,none':'<changelang language="Chinese"/>',
     'change language,English,none':'<changelang language="English"/>',
     'bring,newspaper,none':'<bring object="newspaper"/>',
     'bring,juice,none':'<bring object="juice"/>',
     'bring,socks,none':'<bring object="socks"/>',
     'bring,shoes,none':'<bring object="shoes"/>',
     'activate,music,none':'<switch action="activate" thing="music"/>',
     'deactivate,music,none':'<switch action="deactivate" thing="music"/>',
     'activate,lights,none':'<switch action="activate" thing="lights"/>',
     'deactivate,lights,none':'<switch action="deactivate" thing="lights"/>',
     'activate,lights,kitchen':'<switchwhere action="activate" thing="lights" where="kitchen"/>',
     'deactivate,lights,kitchen':'<switchwhere action="deactivate" thing="lights" where="kitchen"/>',
     'activate,lights,washroom':'<switchwhere action="activate" thing="lights" where="washroom"/>',
     'deactivate,lights,washroom':'<switchwhere action="deactivate" thing="lights" where="washroom"/>',
     'activate,lights,bedroom':'<switchwhere action="activate" thing="lights" where="bedroom"/>',
     'deactivate,lights,bedroom':'<switchwhere action="deactivate" thing="lights" where="bedroom"/>',
     'activate,lamp,none':'<switch action="activate" thing="lamp"/>',
     'deactivate,lamp,none':'<switch action="deactivate" thing="lamp"/>',
     'increase,volume,none':'<change what="volume" how="increase"/>',
     'decrease,volume,none':'<change what="volume" how="decrease"/>',
     'increase,heat,none':'<change what="heat" how="increase"/>',
     'decrease,heat,none':'<change what="heat" how="decrease"/>',
     'increase,heat,kitchen':'<changewhere what="heat" how="increase" where="kitchen"/>',
     'decrease,heat,kitchen':'<changewhere what="heat" how="decrease" where="kitchen"/>',
     'increase,heat,washroom':'<changewhere what="heat" how="increase" where="washroom"/>',
     'decrease,heat,washroom':'<changewhere what="heat" how="decrease" where="washroom"/>',
     'increase,heat,bedroom':'<changewhere what="heat" how="increase" where="bedroom"/>',
     'decrease,heat,bedroom':'<changewhere what="heat" how="decrease" where="bedroom"/>',
     }

Sets = ['train','test','valid']
try:
     os.makedirs(outdir)
except OSError:
     pass
dbf = open(os.path.join(outdir,'database.cfg'),'w')

for set in Sets:
     f = csv.reader(open(os.path.join(csvdir, '%s_data.csv' % set),'r'))

     header = f.next()
     nrfields = len(header)

     for i in range(nrfields):
         if len(header[i])==0:
             header[i] = 'nr'
         vars()[header[i]] = []
     vars()['task']=[]
     for j in f:
         key = '%s,%s,%s' % (j[4],j[5],j[6])
         task.append(cvt[key])
         for i in range(nrfields):
             vars()[header[i]].append(j[i])

     Speakers = Counter(speakerId)
     setf = open(os.path.join(outdir,set+'.cfg'),'w')
     setf.write('[' + set + ']\ndatasections = ')
     for spk in Speakers.keys():
          try:
               os.makedirs(os.path.join(outdir,spk))
          except OSError:
               pass

          setf.write(spk + ' ')
          wavscp = open(os.path.join(outdir,spk,'wav.scp'),'w')
          taskscp = open(os.path.join(outdir,spk,'tasks'),'w')
          for i,e in enumerate(speakerId):
               if e == spk:
                    wavscp.write(set + '_' + nr[i] + ' ' + finaldir + path[i] + '\n')
                    taskscp.write(set + '_' + nr[i] + ' ' + task[i] + '\n')

          dbf.write('['+spk+']\n')
          dbf.write('audio = ' + finaldir + 'assist/' + spk + '/wav.scp\n')
          dbf.write('features = /esat/spchdisk/scratch/hvanhamm/FluentSpeechCommands/mfcc/' + spk + '\n')
          dbf.write('tasks = ' + finaldir + 'assist/' + spk + '/tasks\n\n')
     setf.write('\n')



Objects = Counter(object)



