data_file=r'/home/siddarth.jha@npci.org.in/Documents/Training/Data/SMSSpamCollection.txt'
f=open(data_file,"r")
target=[]
sms=[]

for line in f:
    line = line.strip()

    if line == "":
        continue

    if line[0:4] == 'spam':
        sms.append(line.split('spam')[1].strip())
        target.append('spam')
    
    if line[0:3]=="ham":
        sms.append(line.split('ham')[1].strip())
        target.append("ham")

f.close()
