# Copy-paste from: http://blog.dlfer.xyz/post/2016-10-27-smith-normal-form/

def dims(M):
 num_righe=len(M)
 num_colonne=len(M[0])
 return (num_righe,num_colonne)

def MinAij(M,s):
 num_righe, num_colonne=dims(M)
 ijmin=[s,s]
 valmin=max( max([abs(x) for x in M[j][s:]]) for j in range(s,num_righe) )
 for i in (range(s,num_righe)):
  for j in (range(s,num_colonne)):
   if (M[i][j] != 0 ) and (abs(M[i][j]) <= valmin) :
    ijmin = [i,j]
    valmin = abs(M[i][j])
 return ijmin

def IdentityMatrix(n):
 res=[[0 for j in range(n)] for i in range(n)]
 for i in range(n):
  res[i][i] = 1
 return res

def display(M):
 r=""
 for x in M:
  r += "%s\n" % x
 return r +""

def swap_rows(M,i,j):
 tmp=M[i]
 M[i]=M[j]
 M[j]=tmp

def swap_columns(M,i,j):
 num_of_columns=len(M)
 for x in range(num_of_columns):
  tmp=M[x][i]
  M[x][i] = M[x][j]
  M[x][j] = tmp

def add_to_row(M,x,k,s):
 num_righe,num_colonne=dims(M)
 for tmpj in range(num_colonne):
  M[x][tmpj] += k * M[s][tmpj]

def add_to_column(M,x,k,s):
 num_righe,num_colonne=dims(M)
 for tmpj in range(num_righe):
  M[tmpj][x] += k * M[tmpj][s]

def change_sign_row(M,x):
 num_righe,num_colonne=dims(M)
 for tmpj in range(num_colonne):
  M[x][tmpj] = - M[x][tmpj]

def change_sign_column(M,x):
 num_righe,num_colonne=dims(M)
 for tmpj in range(num_righe):
  M[tmpj][x] = - M[tmpj][x]

def is_lone(M,s):
 num_righe,num_colonne=dims(M)
 if [M[s][x] for x in range(s+1,num_colonne) if M[s][x] != 0] + [ M[y][s] for y in range(s+1,num_righe) if M[y][s] != 0] == []:
  return True
 else:
  return False

def get_nextentry(M,s):
  # find and element which is not divisible by M[s][s]
  num_righe,num_colonne=dims(M)
  for x in range(s+1,num_righe):
   for y in range(s+1,num_colonne):
    if M[x][y] % M[s][s]  != 0:
     return (x,y)
  return None

def Smith(M):
 num_righe,num_colonne=dims(M)
 L = IdentityMatrix(num_righe)
 R = IdentityMatrix(num_colonne)
 maxs=min(num_righe,num_colonne)
 for s in range(maxs):
  # print ("step %s/%s\n" % (s+1,maxs))
  # print "M:", display(M)
  while not is_lone(M,s):
   i,j = MinAij(M,s) # the non-zero entry with min |.|
   swap_rows(M,s,i)
   swap_rows(L,s,i)
   swap_columns(M,s,j)
   swap_columns(R,s,j)
   for x in range(s+1,num_righe):
    if M[x][s] != 0:
     k = M[x][s] // M[s][s]
     add_to_row(M,x,-k,s)
     add_to_row(L,x,-k,s)
   for x in range(s+1,num_colonne):
    if M[s][x] != 0:
     k = M[s][x] // M[s][s]
     add_to_column(M,x,-k,s)
     add_to_column(R,x,-k,s)
   if is_lone(M,s):
    res=get_nextentry(M,s)
    if res:
     x,y=res
     add_to_row(M,s,1,x)
     add_to_row(L,s,1,x)
    else:
     if M[s][s]<0:
      change_sign_row(M,s)
      change_sign_row(L,s)
 return L,R

