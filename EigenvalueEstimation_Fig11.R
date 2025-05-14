library(MASS)
library(nlshrink)
library(here)
library(rstudioapi)
library(tools)


data_path = file_path_as_absolute(".")

d = 1000
c = 10

dist = 10
N = 100

sig2 = 1

n = ceiling(d/c)
PopEV = c(rep(0.5,times=d%/%2),seq(0.5,0.999999,by=1/d))

X = t(as.matrix(mvrnorm(n, rep(0,d), diag(PopEV))))

#Our method
S = X%*%t(X)/n
eig = eigen(S, symmetric = TRUE)
SampEV = eig$values
print('Sample eigenvalues:')
print(SampEV)

z_points = seq(-dist,sig2+dist,by=(sig2+2*dist)/N)+dist*1i
z_points = c(z_points,-dist+seq(0,dist,by=dist/N)*1i)
z_points = c(z_points,sig2+dist+seq(0,dist,by=dist/N)*1i)
plot(Re(z_points), Im(z_points))

LastVal = rep(0,times=length(z_points))
Val = rep(1i,times=length(z_points))
f <- function(v,lam) {
  return (lam/(lam-v))
}
Iterations = 0
while (Iterations < 100 & max(abs(Val-LastVal)) > 10**(-6)){
  Iterations <- Iterations + 1
  v = (1-c*Val)*z_points
  M = outer(v,SampEV,FUN=f)
  LastVal = Val
  Val = rowSums(M)/d
}
if (Iterations==100){
  print("Error: Stieltjes transform estimators not found! Try setting dist larger.")
}
StieltjesEstimators = (Val-1)/z_points
#print(StieltjesEstimators)

p=1
start = seq(0,sig2-10**(-6),by=sig2/d)
StilFunction <- function(z,w){
  return (1/(w-z))
}
objective <- function(w){
  A = outer(z_points,w,FUN=StilFunction)
  Diff = StieltjesEstimators - rowSums(A)/d
  return (sum(abs(Diff)**(2*p)))
}

gradHelpFunction<-function(w,z){
  return (1/(w-z)**2)
}
gradient <- function(w){
  A = outer(z_points,w,FUN=StilFunction)
  Diff = StieltjesEstimators - rowSums(A)/d
  Prod = Diff**(p-1)*Conj(Diff)**p
  B = outer(w,z_points,FUN=gradHelpFunction)
  ProdMatr = t(matrix(rep(Prod, d), nrow = length(Prod), ncol = d))
  return (2*p/d*Re(rowSums(ProdMatr*B)))
}


Estimator = nlminb(start, objective, gradient = gradient, scale = 1, control = list(), lower = 0, upper = sig2)
Estimator = sort(Estimator$par)

output_file_name = paste("EV_Estimation_d=",d,"_c=",c,".csv",sep="")
output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",output_file_name,sep='/')
write.csv(as.data.frame(Estimator),output_file_path, row.names = FALSE)

print('Our estimator')
print(Estimator)

#Ledoit-Wolf method
Estimator_LedoitWolf = tau_estimate(t(X), k = 0, method = "nlminb", control = list())

output_file_name = paste("EV_Estimation_LedoitWolf_d=",d,"_c=",c,".csv",sep="")
output_file_path = paste(dirname(rstudioapi::getSourceEditorContext()$path),"data",output_file_name,sep='/')
write.csv(as.data.frame(Estimator_LedoitWolf),output_file_path, row.names = FALSE)

print('Ledoit-Wolf estimator')
print(Estimator_LedoitWolf)