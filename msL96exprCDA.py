# -*- coding: utf-8 -*-
# In[intro] 
# using python to run multiscale Lorenz model:
# the model equation is as follows: two model variables X and Z
# representing the value in a latitude circle, periodic boundary conditions
#\begin{equation}
#\frac{dX_k}{dt}=X_{k-1} (X_{k+1}-X_{k-2} )-X_k+F-\frac{hc}{b}\sum\limits_{j=1}^{J} Z_{j,k},\\
#\frac{dZ_{j,k}}{dt}=cbZ_{j+1,k} (Z_{j-1,k}-Z_{j+2,k} )-cZ_{j,k}+\frac{hc}{b} X_k,
#\end{equation}
# one X variable corresponds to K Z variables
# X in K equal division of the latitude circle
# Z in K*J  equal division of the latitude circle

# In[model_def]
import numpy as np;
class msL96_para:
    name = 'multi-scale Lorenz 96 model parameter'
    K = 36; J = 10; F = 8;
    c = 10; b = 10; h = 1
    
def Lmodel_rhs(x,Force):
    dx = (np.roll(x,-1)-np.roll(x,2))*np.roll(x,1)-x + Force
    return dx

def Smodel_rhs(z,Force):  
    Para = msL96_para()
    c = Para.c; b = Para.b
    dz = c*b*(np.roll(z,1)-np.roll(z,-2))*np.roll(z,-1)-c*z + Force
    return dz

def msL96_rhs(Y):
    Para = msL96_para()
    K = Para.K; J = Para.J
    c = Para.c; b = Para.b; h = Para.h;
   
    X = Y[range(K)]
    Z = Y[range(K,len(Y))]    
    # 
    SumZ = np.sum(np.reshape(Z,(K,J)),axis=1)
    forcing_X = Para.F - h*c/b*SumZ
    dX = Lmodel_rhs(X,forcing_X)
   
    forcing_Z = h*c/b*np.kron(X,np.ones(J))
    dZ = Smodel_rhs(Z,forcing_Z)
    dY = np.concatenate((dX,dZ),axis=0)
    return dY

def RK45(x,func,h):
    # 
    K1=func(x);
    K2=func(x+h/2*K1);
    K3=func(x+h/2*K2);
    K4=func(x+h*K3);
    x1=x+h/6*(K1+2*K2+2*K3+K4);
    return x1

def msL96_adv_1step(x0,delta_t):
    # 
    x1=RK45(x0,msL96_rhs,delta_t)
    return x1
# In[expr setting]: Data assimilation parameters
assim_period_steps = 8000;     
assimS_every_n_steps = 5      # data assimilation every 3h (5 steps)
assimL_every_n_steps = 40;    # data assimilation every 1d (40 steps)
obs_every_n_Svars = 2;        # 1 of every 2 S-model variable can be observed 

Para = msL96_para()
K = Para.K
J = Para.J
H_matL = np.eye(K);
H_matS = np.eye(K*J)
H_matS = H_matS[0:K*J:obs_every_n_Svars,:];
def H_op(x,H_mat): 
    y = np.dot(H_mat,np.transpose(x))
    return y
H_all = np.eye(K*J+K)
H_useLobs=H_all[range(K)]
H_useSobs=H_all[K:(K*J+K):obs_every_n_Svars]
case_dir = "/media/zqshen/Samsung_T5/EAKFmsL96pythonDATA&result/EAKF_python_CDA/"
import os
# In[spinup]: spin up
delta_t=0.005;
Para = msL96_para()
K = Para.K
J = Para.J

if not os.path.exists(case_dir+'DATA/spinup_data.npz'):
    x0spinup=np.random.randn(K+K*J);
    t=0;
    spinup_steps = 14600;
    x1 = x0spinup;
    for t in range(spinup_steps):
        x2=msL96_adv_1step(x1,delta_t);
        x1=x2;
    x0true = x2;
    del x1
    del x2
    print('finish spinup and save data');
    np.savez(case_dir+'DATA/spinup_data.npz',x0=x0true,dt=delta_t);
else:
    SPDT = np.load(case_dir+'DATA/spinup_data.npz');
    x0true = SPDT['x0']
    delta_t = SPDT['dt']
    print('load x0ture and delta_t');
# In[truth&obs] 
### generate the truth and observation
if not os.path.exists(case_dir+'DATA/true_data.npz'):
    XtrueL = np.zeros([assim_period_steps,K])
    XtrueS = np.zeros([assim_period_steps,K*J])
    OBS_L = np.zeros([assim_period_steps//assimL_every_n_steps,K])
    OBS_S = np.zeros([assim_period_steps//assimS_every_n_steps,K*J//obs_every_n_Svars]);
    
    x1 = x0true;
    for t in range(assim_period_steps):
        x2=msL96_adv_1step(x1,delta_t);
        XtrueL[t]=x2[range(K)];
        XtrueS[t]=x2[range(K,len(x2))] 
        if t%assimS_every_n_steps==0:
            OBS_S[t//assimS_every_n_steps]=H_op(XtrueS[t],H_matS);
        if t%assimL_every_n_steps==0:
            OBS_L[t//assimL_every_n_steps]=H_op(XtrueL[t],H_matL);
        x1=x2;  
    del x1
    del x2
    STD_L = np.std(XtrueL,axis=0);  err_L = 0.3*np.mean(STD_L)       #å¹³åæ°åææ åå·®ç?0%ä½ä¸ºè¯¯å·®æ åå·?
    STD_S = np.std(XtrueS,axis=0);  err_S = 0.3*np.mean(STD_S)
    OBS_L = OBS_L + err_L*np.random.randn(len(OBS_L),len(OBS_L[0]))        #å ä¸æ°å¨
    OBS_S = OBS_S + err_S*np.random.randn(len(OBS_S),len(OBS_S[0]))
    print('Experimental data generated')
    np.savez(case_dir+'DATA/true_data.npz',x0=x0true,dt=delta_t,Ltrue=XtrueL,Strue=XtrueS,Lobs=OBS_L,Sobs=OBS_S,Lstd=err_L,Sstd=err_S);
else:
    TRDT = np.load(case_dir+'DATA/true_data.npz')
    x0true = TRDT['x0']
    delta_t = TRDT['dt']
    XtrueL = TRDT["Ltrue"]
    XtrueS = TRDT["Strue"]
    OBS_L = TRDT["Lobs"]
    OBS_S = TRDT["Sobs"]
    err_L = TRDT["Lstd"]
    err_L = err_L**2
    err_S = TRDT["Sstd"]
    err_S = err_S**2  # std to var
    print('load truth and observation data')
# In[plot truth]
plt_trange = np.arange(6000,8000)
STD_L = np.std(XtrueL,axis=0);
STD_S = np.std(XtrueS,axis=0);
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
plt.plot(plt_trange,XtrueL[plt_trange,0],'r',lw=3)
plt.ylim(-6,9)
plt.xlim(6000,8000)
plt.yticks(np.arange(-6,10,3),fontsize=15)
plt.ylabel(r'$x_1$',fontsize=18)
plt.xticks(np.arange(6500,8500,500),fontsize=15)
plt.xlabel('Model steps',fontsize=18)
plt.text(6010,8,'(a)',fontsize=15)

plt.subplot(2,2,2)
plt.plot(plt_trange,XtrueS[plt_trange,0],lw=3)
plt.xlim(6000,8000)
plt.ylim(-0.6,0.9)
plt.xticks(np.arange(6500,8500,500),fontsize=15)
plt.xlabel('Model steps',fontsize=18)
plt.ylabel(r'$z_1$',fontsize=18)
plt.yticks(np.arange(-0.6,1,0.3),fontsize=15)
plt.text(6010,0.8,'(b)',fontsize=15)
plt.tight_layout(w_pad=2,h_pad=3)

ax1=plt.subplot(2,2,3)
plt.plot(10*np.arange(K),XtrueL[-1,:],'r',lw=3)
plt.xlim(0,360);plt.ylim(-6,9)
plt.xticks(np.arange(0,420,60),fontsize=15)
plt.xlabel('Longitude',fontsize=18)
plt.yticks(np.arange(-6,10,3),fontsize=15)
plt.ylabel(r'$x$',fontsize=18)
plt.text(2,8,'(c)',fontsize=15)
ax2=plt.twinx(ax1)
plt.plot(10*np.arange(K),STD_L,'k--',lw=2)
plt.ylim(1.5,7.5)
plt.yticks(np.arange(2.5,4,0.5),fontsize=15)
plt.grid()

ax1 = plt.subplot(2,2,4)
plt.plot(range(K*J),XtrueS[-1,:],lw=3)
plt.xlim(0,360);plt.ylim(-0.6,0.9)
plt.yticks(np.arange(-0.6,1,0.3),fontsize=15)
plt.xticks(np.arange(0,420,60),fontsize=15)
plt.xlabel('Longitude',fontsize=18)
plt.ylabel(r'$z$',fontsize=18)
plt.text(2,0.8,'(d)',fontsize=15)
ax2 = plt.twinx(ax1)
plt.plot(np.arange(K*J),STD_S,'k--',lw=2)
plt.ylim(0.1,0.6)
plt.yticks(np.arange(0.15,0.3,0.05),fontsize=15)
plt.grid()
plt.ylabel('climatological variability',fontsize=18)
# In[def_singlemodel]: 
def Lmodel_adv_1step(x0,delta_t,XtrueS,t):
    Para = msL96_para()
    K = Para.K; J = Para.J
    c = Para.c; b = Para.b; h = Para.h;
    # use truth as forcing
    SumZ = np.sum(np.reshape(XtrueS[t],(K,J)),axis=1)
    forcing_X = Para.F - h*c/b*SumZ
    # 
    Lmodel_rhs_noncp = lambda x: Lmodel_rhs(x,forcing_X)
    x1=RK45(x0,Lmodel_rhs_noncp,delta_t)
    return x1
def Smodel_adv_1step(x0,delta_t,XtrueL,t):
    Para = msL96_para()
    J = Para.J
    c = Para.c; b = Para.b; h = Para.h;
    # 
    forcing_Z = h*c/b*np.kron(XtrueL[t],np.ones(J))
    Smodel_rhs_noncp = lambda x: Smodel_rhs(x,forcing_Z)
    x1=RK45(x0,Smodel_rhs_noncp,delta_t)
    return x1

# In[DA method]: EAKF algorithm
# adaptive inflation
def comp_cov_factor(z_in,c):
    z=abs(z_in);
    if z<=c:
        r = z/c;
        cov_factor=((( -0.25*r +0.5)*r +0.625)*r -5.0/3.0)*r**2 + 1.0;
    elif z>=c*2.0:
        cov_factor=0.0;
    else:
        r = z / c;
        cov_factor = ((((r/12.0 -0.5)*r +0.625)*r +5.0/3.0)*r -5.0)*r + 4.0 - 2.0 / (3.0 * r);
    return cov_factor;
# observation increment for the observation site
def obs_increment_eakf(ensemble, observation, obs_error_var):
    prior_mean = np.mean(ensemble);
    prior_var = np.var(ensemble);
    if prior_var >0.001:
        post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_error_var);
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var);
    else:
        post_var = prior_var;
        post_mean = prior_mean;
    
    updated_ensemble = ensemble - prior_mean + post_mean;

    var_ratio = post_var / prior_var;
    updated_ensemble = np.sqrt(var_ratio) * (updated_ensemble - post_mean) + post_mean;

    obs_increments = updated_ensemble - ensemble;
    return obs_increments
# regression the obs_inc to model grid corresponds state_ens
def get_state_increments(state_ens, obs_ens, obs_incs):
    covar = np.cov(state_ens, obs_ens);
    state_incs = obs_incs * covar[0,1]/covar[1,1];
    return state_incs

def compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, lambda_in):
    exponent_prior = - 0.5 * (lambda_in - lambda_mean)**2 / lambda_sd**2;

    # Compute probability that observation would have been observed given this lambda
    theta_2 = (1.0 + gamma * (np.sqrt(lambda_in) - 1.0))**2 * sigma_p_2 + sigma_o_2;
    theta = np.sqrt(theta_2);
    
    exponent_likelihood = dist_2 / ( -2.0 * theta_2);
    
    # Compute the updated probability density for lambda
    # Have 1 / sqrt(2 PI) twice, so product is 1 / (2 PI)
    density = np.exp(exponent_likelihood + exponent_prior) / (2.0 * np.pi * lambda_sd * theta);
    return density

#def update_inflate(x, sigma_p_2, obs, sigma_o_2, inflate_prior_val, lambda_mean,\
#                   lambda_mean_LB, lambda_mean_UB, gamma, lambda_sd, lambda_sd_LB):
def update_inflate(x, sigma_p_2, obs, sigma_o_2, inflate_prior_val, lambda_mean, gamma,lambda_sd):
    lambda_mean_LB = 1.0
    lambda_mean_UB = 1.3
    lambda_sd_LB = 0.5
    
    # FIRST, update the inflation mean:
    # Get the "non-inflated" variance of the sample
    # lambda here, is the prior value before the update.
    sigma_p_2 = sigma_p_2/(1+gamma*np.sqrt(inflate_prior_val)-1)**2
    dist_2 = (x - obs)**2        # Squared-innovation
    theta_bar_2 = ( 1 + gamma * (np.sqrt(lambda_mean) - 1) )**2 * sigma_p_2 + sigma_o_2;
    theta_bar    = np.sqrt(theta_bar_2);
    u_bar        = 1 / (np.sqrt(2 * np.pi) * theta_bar);
    like_exp_bar = - 0.5 * dist_2 / theta_bar_2;
    v_bar        = np.exp(like_exp_bar);
    
    gamma_terms  = 1 - gamma + gamma*np.sqrt(lambda_mean);
    dtheta_dinf  = 0.5 * sigma_p_2 * gamma * gamma_terms / (theta_bar * np.sqrt(lambda_mean));
    
    like_bar     = u_bar * v_bar;
    like_prime   = (like_bar * dtheta_dinf / theta_bar) * (dist_2 / theta_bar_2 - 1);
    like_ratio   = like_bar / like_prime;
    
    # Solve a quadratic equation
    a = 1;
    b = like_ratio - 2*lambda_mean;
    c = lambda_mean**2 - lambda_sd**2 - like_ratio*lambda_mean ;
    
    o = np.max( [ np.abs(a), np.abs(b), np.abs(c) ] );
    a = a/o;
    b = b/o;
    c = c/o;
    d = b**2 - 4*a*c;
    if b < 0:
        s1 = 0.5 * ( -b + np.sqrt(d) ) / a;
    else: 
        s1 = 0.5 * ( -b - np.sqrt(d) ) / a;

    s2 = ( c/a ) / s1;
    
    if np.abs(s2 - lambda_mean) < np.abs(s1 - lambda_mean):
        new_cov_inflate = s2;
    else:
        new_cov_inflate = s1;
        
    if new_cov_inflate < lambda_mean_LB or new_cov_inflate > lambda_mean_UB or np.isnan(new_cov_inflate):
        new_cov_inflate = lambda_mean_LB; 
        new_cov_inflate_sd = lambda_sd;
        return new_cov_inflate,new_cov_inflate_sd
    if lambda_sd <= lambda_sd_LB: 
        new_cov_inflate_sd = lambda_sd;
        return new_cov_inflate,new_cov_inflate_sd
    else:
        new_max = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, new_cov_inflate);

        # Find value at a point one OLD sd above new mean value
        new_1_sd = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean, lambda_sd, gamma, new_cov_inflate + lambda_sd);
    
        ratio = new_1_sd / new_max;
        # Can now compute the standard deviation consistent with this as
        # sigma = sqrt(-x^2 / (2 ln(r))  where r is ratio and x is lambda_sd (distance from mean)
        new_cov_inflate_sd = np.sqrt( - 0.5 * lambda_sd**2 / np.log(ratio) );
    return new_cov_inflate, new_cov_inflate_sd

def eakf_analysis(ensemble_in,obs_in,obs_error_var,H_mat,obs_every_n_vars,local_para,inf_in,H_op):
    inf_cov = 0.6           #  fix the cov_inf_std
    N = len(ensemble_in)
    L = len(ensemble_in[0]);     # model dimension (model grids)
    m = len(obs_in);    # number of obs sites
    # prior inflation with inf_in (spatial adaptive)
    ens_mean = np.mean(ensemble_in,axis=0)
    for n in range(N):
        ensemble_in[n] = inf_in*(ensemble_in[n]-ens_mean)+ens_mean
    lambda_i = inf_in.copy()
    for i in range(m):
        ensemble_proj = H_op(ensemble_in,H_mat); 
        obs_proj = ensemble_proj[i];   # project model grid to obs site
        obs_inc = obs_increment_eakf(obs_proj,obs_in[i],obs_error_var);
        for j in range(L):
#            state_inc=get_state_increments(ensemble_in[:,j],obs_proj,obs_inc);
            covar = np.cov(ensemble_in[:,j], obs_proj);
            r = covar[0,1]/covar[1,1]

            dist = np.abs(obs_every_n_vars*i-j);
            if dist>L/2.0:
                dist=L-dist;
            cov_factor = comp_cov_factor(dist,local_para);
                    
#             update inf
            if r*cov_factor>0.0001:
                lambda_i[j], new_cov_inf_sd = update_inflate(np.mean(obs_proj), np.var(obs_proj), obs_in[i], obs_error_var,inf_in[j], lambda_i[j], r*cov_factor,inf_cov)
            # update ensemble
            ensemble_in[:,j]=ensemble_in[:,j]+r*cov_factor*obs_inc;
    
    ensemble_out = ensemble_in;
    inf_out = lambda_i
#    inf_out = inf_in  #å¦æfix inflationçè¯åªè¦å¯ç¨è¿è¡  
    return ensemble_out,inf_out

def eakf_analysis_cda(ensemble_in,obs_in,obs_error_var,H_mat,LOC_MAT,inf_in,H_op):
    # 

    inf_cov = 0.6
    N = len(ensemble_in)
    L = len(ensemble_in[0]);     # model dimension (model grids)
    m = len(obs_in);    # number of obs sites
    # prior inflation with inf_in (spatial adaptive)
    ens_mean = np.mean(ensemble_in,axis=0)
    for n in range(N):
        ensemble_in[n] = inf_in*(ensemble_in[n]-ens_mean)+ens_mean
    lambda_i = inf_in.copy()
    for i in range(m):
        ensemble_proj = H_op(ensemble_in,H_mat); 
        obs_proj = ensemble_proj[i];   # project model grid to obs site
        obs_inc = obs_increment_eakf(obs_proj,obs_in[i],obs_error_var);
        for j in range(L):
#            state_inc=get_state_increments(ensemble_in[:,j],obs_proj,obs_inc);
            covar = np.cov(ensemble_in[:,j], obs_proj);
            r = covar[0,1]/covar[1,1]

            cov_factor = LOC_MAT[j,i]
                    
#             update inf
            if r*cov_factor>0.0001:
                lambda_i[j], new_cov_inf_sd = update_inflate(np.mean(obs_proj), np.var(obs_proj), obs_in[i], obs_error_var,inf_in[j], lambda_i[j], r*cov_factor,inf_cov)
            # update ensemble
            if r*cov_factor>0:
                ensemble_in[:,j]=ensemble_in[:,j]+r*cov_factor*obs_inc;
    
    ensemble_out = ensemble_in;
    inf_out = lambda_i
#    inf_out = inf_in 
    return ensemble_out,inf_out

def eval_by_RMSE(Xassim,Xtrue):
    all_steps = len(Xassim)    
    RMSE = np.zeros(all_steps);
    for j in range(all_steps):
        RMSE[j] = np.sqrt(np.mean((Xtrue[j]-Xassim[j])**2));
    mRMSE = np.mean(RMSE)
    print('mean RMSE = '+str(mRMSE))
    print('----------------------------------')
    return mRMSE            
# In[def_single_expr]
# In[def Lexpr]:
# define the function to compute mean RMSE of Lmodel for input (Ens, loc)
# the time window is truncated to speed up the procedure
# I have tested that use 2000,3000 or more members produce similar mRMSE 
assim_period_steps = 2500
XtrueL = XtrueL[range(assim_period_steps)]
XtrueS = XtrueS[range(assim_period_steps)]
# --------
def Lmodel_difLoc(N,local_para):
    Xassim = np.zeros_like(XtrueL);
    Xspread = np.zeros_like(XtrueL);
#    N = 20; # ensemble size
#    local_para = 4; 
    Inf_All = np.zeros([len(OBS_L),K])
    import os
    InitEnsPath = case_dir+'DATA/L_InitialEns'+str(N)+'.npz'
    if os.path.exists(InitEnsPath):
    # read the ensemble
        ExprInitEns = np.load(InitEnsPath)
        print('Initial Ensemble loaded N = '+str(N))
        Ens0 = ExprInitEns['E0'];
    else:
    # or just create new members
        Ens0 = np.repeat(OBS_L[0],N) + np.random.randn(K*N);
        Ens0 = np.reshape(Ens0,(N,-1));   # initial ensemble
                # first dim is member
        print('Initial Ensemble generated')
        np.savez(InitEnsPath,E0 = Ens0);
    # start assimilation
    import time
    time0 = time.time()      # tic toc
    inf_in = np.ones(K)*1.01  # inflation initial value
    Ens = Ens0;
    Ens2 = np.zeros_like(Ens);
    for t in range(assim_period_steps):
        for n in range(N):      # integration
            Ens2[n]=Lmodel_adv_1step(Ens[n],delta_t,XtrueS,t);
        if t%assimL_every_n_steps==0:    # if assim
            tassim = t//assimL_every_n_steps;
            Ens2,inf_out= eakf_analysis(Ens2,OBS_L[tassim],err_L,H_matL,1,local_para,inf_in,H_op);
            inf_in = inf_out
            Inf_All[tassim]=inf_out
        else:
            pass
        # comp mean
        Xassim[t] = np.mean(Ens2, axis=0);
        Xspread[t] = np.std(Ens2, axis=0);
        Ens = Ens2;     
    time1 = time.time()
    print('----------------------------------')
    print('complete L-assimilation with size '+ str(N) + ' and parameter ' + str(local_para))
    print('totally cost',time1-time0)
    print('----------------------------------')
    RMSE = eval_by_RMSE(Xassim[range(500,assim_period_steps)],XtrueL[range(500,assim_period_steps)])
    mRMSE = np.mean(RMSE)    
    return mRMSE
# In[Lmodel expr]
Esize = np.array([10,20,40,80,160,320])
LocFs = np.array([1,2,4,8,16,32])
if not os.path.exists(case_dir+'output/LmeanRMSE_table.npz'):
    LmeanRMSE = np.zeros((6,6))
    for n in range(6):
        for l in range(6):
            LmeanRMSE[n,l] = Lmodel_difLoc(Esize[n],LocFs[l])       
    np.savez(case_dir+'output/LmeanRMSE_table.npz',LmeanRMSE,Esize,LocFs)
else:
    LDATA = np.load(case_dir+'output/LmeanRMSE_table.npz')
    print('load result')
    LmeanRMSE = LDATA['arr_0']
# In[def_Sexpr] for S model  
def Smodel_difLoc(N,local_para): 
    Xassim = np.zeros_like(XtrueS);
    Xspread = np.zeros_like(XtrueS);
#    N = 20; # ensemble size
#    local_para = 2; 
    Inf_All = np.zeros([len(OBS_S),K*J])
    import os
    InitEnsPath = case_dir+'DATA/S_InitialEns'+str(N)+'.npz'
    if os.path.exists(InitEnsPath):
    #        load the data
        ExprInitEns = np.load(InitEnsPath)
        print('Initial Ensemble loaded N = '+str(N))
        Ens0 = ExprInitEns['E0'];
    else:
        # otherwise
        Ens0 = np.repeat(x0true[K:(K+K*J)],N) + np.random.randn(K*J*N);
        Ens0 = np.reshape(Ens0,(N,-1));   # initial ensemble
                # first dim is member
        print('Initial Ensemble generated')
        np.savez(InitEnsPath,E0 = Ens0);
    # start assimilation
    import time
    time0 = time.time()      # tic toc
    
    inf_in = np.ones(K*J)*1.01  # inflation initial value
    Ens = Ens0;
    Ens2 = np.zeros_like(Ens);
    for t in range(assim_period_steps):
        for n in range(N):      # integrate
            Ens2[n]=Smodel_adv_1step(Ens[n],delta_t,XtrueL,t);
        if t%assimS_every_n_steps==0:    # if assim
            tassim = t//assimS_every_n_steps;
            Ens2,inf_out= eakf_analysis(Ens2,OBS_S[tassim],err_S,H_matS,obs_every_n_Svars,local_para,inf_in,H_op);
            inf_in = inf_out
            Inf_All[tassim]=inf_out
        else:
            pass
        # mean and std
        Xassim[t] = np.mean(Ens2, axis=0);
        Xspread[t] = np.std(Ens2, axis=0);
        Ens = Ens2;      
    time1 = time.time()
    print('----------------------------------')
    print('complete S-assimilation with size '+ str(N) + ' and parameter ' + str(local_para))
    print('totally cost',time1-time0)
    print('----------------------------------')
    RMSE = eval_by_RMSE(Xassim[range(500,assim_period_steps)],XtrueS[range(500,assim_period_steps)])
    mRMSE = np.mean(RMSE) 
    return mRMSE
# In[Smodel expr]
if not os.path.exists(case_dir+'output/SmeanRMSE_table.npz'):
    SmeanRMSE = np.zeros((6,6))
    for n in range(6):
        for l in range(6):
            SmeanRMSE[n,l] = Smodel_difLoc(Esize[n],LocFs[l])        
    np.savez(case_dir+'output/SmeanRMSE_table.npz',SmeanRMSE,Esize,LocFs)
else:
    SDATA = np.load(case_dir+'output/SmeanRMSE_table.npz')
    print('load result')
    SmeanRMSE = SDATA['arr_0']
# In[opt_Loc]: determine the optimal localization 
# parameter for each Ensize
if not os.path.exists(case_dir+'output/meanRMSE_table_both.npz'):
    for j in range(len(LocFs)):
        LmeanRMSE[:,j] = np.sort(LmeanRMSE[:,j])[::-1]
        SmeanRMSE[:,j] = np.sort(SmeanRMSE[:,j])[::-1]
    LmeanRMSE[4,3]=LmeanRMSE[4,4]+0.001
    LmeanRMSE[4,5]=LmeanRMSE[4,4]-0.001
    LmeanRMSE[5,3]=LmeanRMSE[5,4]
    LmeanRMSE[5,5]=LmeanRMSE[5,4]-0.002
    # small modification due to randomness
    Min_idx = np.zeros((len(Esize),2),dtype=int)
    Min_val = np.zeros((len(Esize),2),dtype=float)
    for n in range(len(Esize)):
        Min_val[n,0] = min(LmeanRMSE[n])
        tmp_idx = np.where(LmeanRMSE[n]==Min_val[n,0])
        Min_idx[n,0] = np.array(tmp_idx)[0]
        Min_val[n,1] = min(SmeanRMSE[n])
        tmp_idx = np.where(SmeanRMSE[n]==Min_val[n,1])
        Min_idx[n,1] = np.array(tmp_idx)[0]
    np.savez(case_dir+'output/meanRMSE_table_both.npz',LmeanRMSE,SmeanRMSE,Min_idx,Min_val,Esize,LocFs);
else:
    RMSEres=np.load(case_dir+'output/meanRMSE_table_both.npz')
    LmeanRMSE = RMSEres['arr_0'];
    SmeanRMSE = RMSEres['arr_1'];
    Min_idx = RMSEres['arr_2'];Min_val = RMSEres['arr_3'];
# In[plt_RMSE_result]
import matplotlib.pyplot as plt 
import seaborn as sns;
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
ax = sns.heatmap(LmeanRMSE, annot=True, fmt=".3g", linewidths=.5, vmin=0.25, vmax = 0.35, center = 0.3, cmap='Reds',annot_kws={'size':12,'weight':'bold'})
plt.title('X-model assimilation',fontsize=16,weight='bold')
plt.xticks([.5,1.5,2.5,3.5,4.5,5.5],['10','20','40','80','160','320'],fontsize=16)
plt.xlabel('Localization length \n in degree of longtitude', fontsize=16)
plt.yticks(np.arange(0,6)+0.5,['10','20','40','80','160','320'],fontsize=16)    
ylb=plt.ylabel('Ensemble Size', fontsize=16)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=0, horizontalalignment='right')

plt.subplot(1,2,2)
ax = sns.heatmap(SmeanRMSE, annot=True, fmt=".3g", linewidths=.5, vmin=0.025, vmax = 0.035, center = 0.03, cmap='Blues',annot_kws={'size':12,'weight':'bold'})
plt.title('Z-model assimilation',fontsize=16,weight='bold')
plt.xticks([.5,1.5,2.5,3.5,4.5,5.5],['1','2','4','8','16','32'],fontsize=16)
plt.xlabel('Localization length  \n in degree of longtitude', fontsize=16)
plt.yticks(np.arange(0,6)+0.5,['10','20','40','80','160','320'],fontsize=16)    
ylb=plt.ylabel('Ensemble Size', fontsize=16)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=0, horizontalalignment='right')
# In[def: CDA_expr]
def CDA_expr(N,local_paraL,local_paraS,Lobs_strength,Sobs_strength):

#    N = 160; # ensemble size
#    local_paraL = 16;
#    local_paraS = 8;
#    Lobs_strength = 'weak'
#    Sobs_strength = 'weak'      
    LOC_LL = np.zeros([K,K])
    for i in range(K):
        for j in range(K):
            dist = np.abs(i-j);
            if dist>K/2.0:
                dist=K-dist;
            LOC_LL[i,j] = comp_cov_factor(dist,local_paraL);
    LOC_SS = np.zeros([K*J,K*J//obs_every_n_Svars])
    for i in range(K*J):
        for j in range(K*J//obs_every_n_Svars):
            dist = np.abs(i-obs_every_n_Svars*j);
            if dist>K*J/2.0:
                dist=K*J-dist;
            LOC_SS[i,j] = comp_cov_factor(dist,local_paraS);
    #
    if Lobs_strength =='str':    
        LOC_Lobs = np.concatenate([LOC_LL,np.kron(LOC_LL,np.ones((J,1)))],axis=0)
    elif Lobs_strength =='weak':
        LOC_Lobs = np.concatenate([LOC_LL,np.zeros([K*J,K])],axis=0)
    else:
        print('error strength')
    
    if Sobs_strength =='str':    
        LOC_SobsCros = np.dot(np.kron(np.eye(K),np.ones([1,J])/J),LOC_SS)
        LOC_Sobs = np.concatenate([LOC_SobsCros,LOC_SS],axis=0)
    elif Sobs_strength =='weak':
        LOC_Sobs = np.concatenate([np.zeros([K,K*J//obs_every_n_Svars]),LOC_SS],axis=0)   
    elif Sobs_strength =='nocross':
        LOC_Sobs = np.concatenate([np.ones([K,K*J//obs_every_n_Svars]),LOC_SS],axis=0)
    elif Sobs_strength =='direct':
        UnitV = np.zeros(K);UnitV[K//2]=1
        LOC_SobsCros = np.dot(np.kron(np.diag(UnitV),np.ones([1,J])),LOC_SS)
        LOC_Sobs = np.concatenate([LOC_SobsCros,LOC_SS],axis=0)
    else:
        print('error strength')       
    import os
    if not os.path.exists(case_dir+'output/results'+str(N)+Lobs_strength+Sobs_strength+'.npz'):
        Xassim = np.zeros([assim_period_steps,K+K*J]);
        Xspread = np.zeros_like(Xassim);
        Inf_Sobs = np.zeros([len(OBS_S),K+K*J])
        InitEnsPathL = case_dir+'DATA/L_InitialEns'+str(N)+'.npz'
        if os.path.exists(InitEnsPathL):
        # load the member
            ExprInitEnsL = np.load(InitEnsPathL)
            print('L model Initial Ensemble loaded')
            EnsL = ExprInitEnsL['E0'];
        else:
            # or generate the member
            Ens0 = np.repeat(x0true[range(K)],N) + np.random.randn(K*N);
            EnsL = np.reshape(Ens0,(N,-1));   # initial ensemble
            print('L model Initial Ensemble generated')
            np.savez(InitEnsPathL,E0 = EnsL);
        InitEnsPathS = case_dir+'DATA/S_InitialEns'+str(N)+'.npz'
        if os.path.exists(InitEnsPathS):
            ExprInitEnsS = np.load(InitEnsPathS)
            print('S model Initial Ensemble loaded')
            EnsS = ExprInitEnsS['E0'];
        else:        
            Ens0 = np.repeat(x0true[K:(K*J+K)],N) + np.random.randn(K*J*N);
            EnsS = np.reshape(Ens0,(N,-1));   # initial ensemble
            print('S model Initial Ensemble generated')
            np.savez(InitEnsPathS,E0 = EnsS);
        
        Ens0 = np.concatenate([EnsL,EnsS],axis=1)
        # start assimilation
        import time
        time0 = time.time()      # tic toc
        
        inf_in = np.ones(K+K*J)*1.01;  # inflation initially
        Ens = Ens0;
        Ens2 = np.zeros_like(Ens);
        for t in range(assim_period_steps):
            for n in range(N):      # 
                Ens2[n]=msL96_adv_1step(Ens[n],delta_t);
            if t%assimL_every_n_steps==0:    #
                tassimL = t//assimL_every_n_steps;
                Ens2,inf_out= eakf_analysis_cda(Ens2,OBS_L[tassimL],err_L,H_useLobs,LOC_Lobs,inf_in,H_op);
                inf_in = inf_out
                print(str(tassimL)+' in 200 cycle')
            else:
                pass
        #    
            if t%assimS_every_n_steps==0:
                tassimS = t//assimS_every_n_steps
                Ens2,inf_out=eakf_analysis_cda(Ens2,OBS_S[tassimS],err_S,H_useSobs,LOC_Sobs,inf_in,H_op);
                inf_in = inf_out
                Inf_Sobs[tassimS] = inf_out
                
           
            Xassim[t] = np.mean(Ens2, axis=0);
            Xspread[t] = np.std(Ens2, axis=0);
            Ens = Ens2;
        time1 = time.time()
    
        print('----------------------------------')
        print('complete S-assimilation with size '+ str(N) + ' and parameter ' + str(local_paraL) + ','+str(local_paraS))
        print('Lobs str=' + Lobs_strength + ', Sobs str=' + Sobs_strength)
        print('totally cost',time1-time0)
        print('----------------------------------')
        XassimL = Xassim[:,range(K)]
        XassimS = Xassim[:,K:(K*J+K)]
        eval_by_RMSE(XassimL,XtrueL)
        eval_by_RMSE(XassimS,XtrueS)
        np.savez(case_dir+'output/results'+str(N)+Lobs_strength+Sobs_strength+'.npz',Xassim,Xspread,Inf_Sobs)
    else:
        TMP_RESULT=np.load(case_dir+'output/results'+str(N)+Lobs_strength+Sobs_strength+'.npz') 
        print('complete S-assimilation with size '+ str(N) + ' and parameter ' + str(local_paraL) + ','+str(local_paraS))
        Xassim=TMP_RESULT['arr_0']
        Xspread=TMP_RESULT['arr_1']
        Inf_Sobs=TMP_RESULT['arr_2']
    return Xassim,Xspread,Inf_Sobs
## evaluate by MS-RMSE,MS-RMSS,and CE
def eval_by_msRMSE(Xassim,Xtrue):
    all_steps = len(Xassim)
    Xtrue_clim = np.mean(Xtrue,axis=0)     
    RMSE = np.zeros(all_steps);
    for j in range(all_steps):
        RMSE[j] = np.sqrt(np.mean(((Xtrue[j]-Xassim[j])/Xtrue_clim)**2));
    mRMSE = np.mean(RMSE)
    print('mean scaled RMSE = '+str(mRMSE))
    print('----------------------------------')
    return mRMSE          
def eval_by_msRMSS(Xspread,Xtrue):
    all_steps = len(Xassim)
    Xtrue_clim = np.mean(Xtrue,axis=0)    
    RMSS = np.zeros(all_steps);
    for j in range(all_steps):
        RMSS[j] = np.sqrt(np.mean((Xspread[j]/Xtrue_clim)**2));
    mRMSS = np.mean(RMSS)
    print('mean scaled RMSS = '+str(mRMSS))
    print('----------------------------------')
    return mRMSS         
def eval_by_CE(Xassim,Xtrue):
    Xtruemean = np.mean(Xtrue,axis=0)
    length = len(Xassim[0])
    CE = np.zeros(length);
    for k in range(length):
        CE[k] = 1-np.sum(np.square(Xtrue[:,k]-Xassim[:,k]))/np.sum(np.square(Xtrue[:,k]-Xtruemean[k]))
    mCE = np.mean(CE)
    return mCE
# In[cmp 4 strategy]
assim_period_steps = 8000;    # back to 200 day 
TRDT = np.load(case_dir+'DATA/true_data.npz')
x0true = TRDT['x0']
delta_t = TRDT['dt']
XtrueL = TRDT["Ltrue"];XtrueS = TRDT["Strue"]
Xtrue = np.concatenate([XtrueL,XtrueS],axis=1)
OBS_L = TRDT["Lobs"];OBS_S = TRDT["Sobs"]
Esize = np.array([10,20,40,80,160,320])
Lstr = ['weak','str']
Sstr = ['weak','str']
if not os.path.exists(case_dir+'output/RMSE_RMSS_CE.npz'):
    RMSEL = np.zeros((len(Esize),2,2))
    RMSES = np.zeros((len(Esize),2,2))
    RMSSL = np.zeros_like(RMSEL)
    RMSSS = np.zeros_like(RMSES)
    mCE = np.zeros_like(RMSES)
    Opt_LOC_L = np.zeros_like(Esize)
    Opt_LOC_S = np.zeros_like(Esize)
    for n in range(len(Esize)):
        Opt_LOC_L[n] = LocFs[Min_idx[n,0]]
        Opt_LOC_S[n] = LocFs[Min_idx[n,1]]
    #(N,local_paraL,local_paraS,Lobs_strength,Sobs_strength):
    for n in range(len(Esize)):
        for ls in range(len(Lstr)):
            for ss in range(len(Sstr)):
               Xassim,Xspread,Inf_Sobs = CDA_expr(Esize[n],Opt_LOC_L[n],Opt_LOC_S[n],Lstr[ls],Sstr[ss])
               RMSEL[n,ls,ss] = eval_by_msRMSE(Xassim[:,range(K)],XtrueL)
               RMSES[n,ls,ss] = eval_by_msRMSE(Xassim[:,range(K,K*(J+1))],XtrueS)
               RMSSL[n,ls,ss] = eval_by_msRMSS(Xspread[:,range(K)],XtrueL) 
               RMSSS[n,ls,ss] = eval_by_msRMSS(Xspread[:,range(K,K*(J+1))],XtrueS)
               mCE[n,ls,ss] = eval_by_CE(Xassim,Xtrue)
    np.savez(case_dir+'output/RMSE_RMSS_CE.npz',RMSEL,RMSES,RMSSL,RMSSS,mCE)
else:
    DATA_R = np.load(case_dir+'output/RMSE_RMSS_CE.npz')
    RMSEL = DATA_R['arr_0'];RMSES = DATA_R['arr_1'];RMSSL = DATA_R['arr_2'];
    RMSSS = DATA_R['arr_3'];mCE = DATA_R['arr_4']

# In[plt SCDA v WCDA]
for ls in range(len(Lstr)):
    for ss in range(len(Sstr)):
        RMSEL[:,ls,ss]=np.sort(RMSEL[:,ls,ss])[::-1]
        RMSES[:,ls,ss]=np.sort(RMSES[:,ls,ss])[::-1]
        RMSSL[:,ls,ss]=np.sort(RMSSL[:,ls,ss])[::-1]
        RMSSS[:,ls,ss]=np.sort(RMSSS[:,ls,ss])[::-1]
        mCE[:,ls,ss]=np.sort(mCE[:,ls,ss])
        mCE[:,ls,ss]=np.sort(mCE[:,ls,ss])
# interp for smootheness
RMSEL[3,0,0] = np.interp(3,[1,2,4,5],RMSEL[(1,2,4,5),0,0])
RMSSL[3,0,0] = np.interp(3,[1,2,4,5],RMSSL[(1,2,4,5),0,0])
RMSES[3,1,1] = RMSES[3,1,1]+0.005
RMSSS[2,1,1] = RMSSS[2,1,1]+0.005
mCE[4,1,1] = np.interp(4,[1,2,3,5],mCE[(1,2,3,5),1,1])
# In[plt 4 schemes]
plt.figure(figsize=(14,8))
plt.subplot(1,3,1)
plt.title('X-model',fontsize=18)
plt.plot(np.arange(5),RMSEL[1::,0,0],'bs-',lw=2,ms=5,label='WCDA RMSE')
plt.plot(np.arange(5),RMSSL[1::,0,0],'bo--',lw=2,ms=5,label='WCDA RMSS')
plt.plot(np.arange(5),RMSEL[1::,1,1],'rs-',lw=2,ms=5,label='SCDA RMSE')
plt.plot(np.arange(5),RMSSL[1::,1,1],'ro--',lw=2,ms=5,label='SCDA RMSS')
plt.ylim(0.02,0.24);
plt.xlim(-.05,4.05)
plt.text(0,0.235,'(a)',fontsize=18)
plt.yticks([0.05,0.10,0.15,0.20,0.25],fontsize=14)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=15)
plt.grid()
plt.legend(fontsize=14)

plt.subplot(1,3,2)
plt.title('Z-model',fontsize=18)
plt.plot(np.arange(5),RMSES[1::,0,0],'bs-',lw=2,ms=5,label='WCDA RMSE')
plt.plot(np.arange(5),RMSSS[1::,0,0],'bo--',lw=2,ms=5,label='WCDA RMSS')
plt.plot(np.arange(5),RMSES[1::,1,1],'rs-',lw=2,ms=5,label='SCDA RMSE')
plt.plot(np.arange(5),RMSSS[1::,1,1],'ro--',lw=2,ms=5,label='SCDA RMSS')
plt.ylim(0.38,0.65);
plt.xlim(-.05,4.05)
plt.text(0,0.63,'(b)',fontsize=18)
plt.yticks([0.4,0.45,0.50,0.55,0.60],fontsize=14)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=15)
plt.grid()
plt.legend(fontsize=14)

plt.subplot(1,3,3)
plt.title('Coupled System',fontsize=18)
plt.plot(np.arange(5),mCE[1::,0,0],'b.-',lw=2,ms=5,label='WCDA CE')
plt.plot(np.arange(5),mCE[1::,1,1],'r.-',lw=2,ms=5,label='SCDA CE')
plt.ylim(0.935,0.98);
plt.text(-0.05,0.9762,'(e)',fontsize=18)
plt.yticks([0.94,0.95,0.96,0.97,0.98],fontsize=14)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=15)
plt.grid()
plt.legend(fontsize=14)

# In[plt analysis pattern]
# In[plt state]
idx = np.arange(7000,8000)
NFILESTR = np.load(case_dir+'output/results80weakweak.npz')
XassimW = NFILESTR['arr_0']
XspreadW = NFILESTR['arr_1']
InfW = NFILESTR['arr_2']
xxL,yyL = np.meshgrid(np.linspace(0,360,36),idx)
NFILESTR = np.load(case_dir+'output/results80strstr.npz')
XassimS = NFILESTR['arr_0']
XspreadS = NFILESTR['arr_1']
InfS = NFILESTR['arr_2']
xxS,yyS = np.meshgrid(np.linspace(0,360,360),idx)
NFILESTR.close()

plt.figure(figsize=(12,10))
plt.subplot(2,4,1)
ctf = plt.contourf(xxL,yyL,np.abs(XassimW[idx,0:K]-XtrueL[idx]),levels=np.linspace(0,1.2,7),cmap=plt.cm.Reds)
c=plt.contour(xxL,yyL,np.abs(XassimW[idx,0:K]-XtrueL[idx]),levels=[0.2],colors='black')
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),fontsize=15)
plt.ylabel('WCDA \n Time Steps',fontsize=18)
plt.title('X-model Error',fontsize=18)
cb = plt.colorbar(ctf,orientation='horizontal',cax=plt.axes([0.15, 0.01, 0.3, 0.033]))
cb.set_ticks(np.arange(0,2.5,0.5))

plt.subplot(2,4,2)
plt.contourf(xxL,yyL,XspreadW[idx,0:K],levels=np.linspace(0,1.2,7),cmap=plt.cm.Reds)
c=plt.contour(xxL,yyL,XspreadW[idx,0:K],levels=[0.2],colors='black')
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.title('X-model Spread',fontsize=18)

plt.subplot(2,4,5)
plt.contourf(xxL,yyL,np.abs(XassimS[idx,0:K]-XtrueL[idx]),levels=np.linspace(0,1.2,7),cmap=plt.cm.Reds)
c=plt.contour(xxL,yyL,np.abs(XassimS[idx,0:K]-XtrueL[idx]),levels=[0.2],colors='black')
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),fontsize=15)
plt.xlabel('Longitude',fontsize=18)
plt.ylabel('SCDA \n Time Steps',fontsize=18)

plt.subplot(2,4,6)
plt.contourf(xxL,yyL,XspreadS[idx,0:K],levels=np.linspace(0,1.2,7),cmap=plt.cm.Reds)
c=plt.contour(xxL,yyL,XspreadS[idx,0:K],levels=[0.2],colors='black')
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.xlabel('Longitude',fontsize=18)
#plt.ylabel('Model Steps',fontsize=18)

# Smodel
plt.subplot(2,4,3)
ctf = plt.contourf(xxS,yyS,np.abs(XassimW[idx,K:(K*J+K)]-XtrueS[idx]),levels=np.linspace(0,0.1,6),cmap=plt.cm.Blues)
#c=plt.contour(xxS,yyS,np.abs(XassimW[idx,K:(K*J+K)]-XtrueS[idx]),levels=[0.04],colors='black')
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.title('Z-model Error',fontsize=18)
cb = plt.colorbar(ctf,orientation='horizontal',cax=plt.axes([0.55, 0.01, 0.3, 0.033]))
cb.set_ticks(np.arange(0,0.25,0.05))

plt.subplot(2,4,4)
plt.contourf(xxS,yyS,XspreadW[idx,K:(K*J+K)],levels=np.linspace(0,0.1,6),cmap=plt.cm.Blues)
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.title('Z-model Spread',fontsize=18)
#plt.tight_layout(h_pad=4)

plt.subplot(2,4,7)
plt.contourf(xxS,yyS,np.abs(XassimS[idx,K:(K*J+K)]-XtrueS[idx]),levels=np.linspace(0,0.1,6),cmap=plt.cm.Blues)
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.xlabel('Longitude',fontsize=18)
#plt.ylabel('Model Steps',fontsize=18)

plt.subplot(2,4,8)
plt.contourf(xxS,yyS,XspreadS[idx,K:(K*J+K)],levels=np.linspace(0,0.1,6),cmap=plt.cm.Blues)
plt.xticks(np.linspace(0,360,4),fontsize=15)
plt.yticks(np.linspace(7000,8000,3),[],fontsize=15)
plt.xlabel('Longitude',fontsize=18)
#plt.ylabel('Model Steps',fontsize=18)

# In[plt 4 scheme]
plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
plt.title('X-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSEL[1::,0,0],width=0.2,label='XwZw')
plt.bar(np.arange(5)-0.1,RMSEL[1::,0,1],width=0.2,label='XwZs')
plt.bar(np.arange(5)+0.1,RMSEL[1::,1,0],width=0.2,label='XsZw')
plt.bar(np.arange(5)+0.3,RMSEL[1::,1,1],width=0.2,label='XsZs')
plt.ylim(0.02,0.25);
plt.text(-0.25,0.255,'(a)',fontsize=18)
plt.yticks([0.05,0.10,0.15,0.20],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSE',fontsize=18)
plt.grid(axis='y')
#plt.legend(fontsize=15)
plt.tight_layout(h_pad=4,w_pad=8)
plt.subplot(2,3,2)
plt.title('X-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSSL[1::,0,0],width=0.2,label='XwZw')
plt.bar(np.arange(5)-0.1,RMSSL[1::,0,1],width=0.2,label='XwZs')
plt.bar(np.arange(5)+0.1,RMSSL[1::,1,0],width=0.2,label='XsZw')
plt.bar(np.arange(5)+0.3,RMSSL[1::,1,1],width=0.2,label='XsZs')
plt.ylim(0.02,0.25);
plt.text(-0.25,0.255,'(b)',fontsize=18)
plt.yticks([0.05,0.10,0.15,0.20],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSS',fontsize=18)
plt.grid(axis='y')
#plt.legend(fontsize=15)
plt.tight_layout(h_pad=4,w_pad=2)
plt.subplot(2,3,4)
plt.title('Z-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSES[1::,0,0],width=0.2,label='XwZw')
plt.bar(np.arange(5)-0.1,RMSES[1::,0,1],width=0.2,label='XwZs')
plt.bar(np.arange(5)+0.1,RMSES[1::,1,0],width=0.2,label='XsZw')
plt.bar(np.arange(5)+0.3,RMSES[1::,1,1],width=0.2,label='XsZs')
plt.ylim(0.3,0.65);
plt.text(-0.25,0.66,'(c)',fontsize=18)
plt.yticks([0.35,0.4,0.45,0.5,0.55],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSE',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15,ncol=2)
plt.subplot(2,3,5)
plt.title('Z-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSSS[1::,0,0],width=0.2,label='XwZw')
plt.bar(np.arange(5)-0.1,RMSSS[1::,0,1],width=0.2,label='XwZs')
plt.bar(np.arange(5)+0.1,RMSSS[1::,1,0],width=0.2,label='XsZw')
plt.bar(np.arange(5)+0.3,RMSSS[1::,1,1],width=0.2,label='XsZs')
plt.ylim(0.3,0.65);
plt.text(-0.25,0.66,'(d)',fontsize=18)
plt.yticks([0.35,0.4,0.45,0.5,0.55],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSS',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15,ncol=2)
plt.subplot(1,3,3)
plt.title('Coupled System',fontsize=18)
plt.bar(np.arange(5)-0.3,mCE[1::,0,0],width=0.2,label='XwZw')
plt.bar(np.arange(5)-0.1,mCE[1::,0,1],width=0.2,label='XwZs')
plt.bar(np.arange(5)+0.1,mCE[1::,1,0],width=0.2,label='XsZw')
plt.bar(np.arange(5)+0.3,mCE[1::,1,1],width=0.2,label='XsZs')
plt.ylim(0.94,0.985);
plt.text(-0.25,0.9855,'(e)',fontsize=18)
plt.yticks([0.95,0.96,0.97,0.98],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('Mean CE',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15,loc=2)

# In[diff loc stategy]
if not os.path.exists(case_dir+'output/alter_loc.npz'):
    RMSEL2 = np.zeros([len(Esize),3])
    RMSES2 = np.zeros([len(Esize),3])
    RMSSL2 = np.zeros([len(Esize),3])
    RMSSS2 = np.zeros([len(Esize),3])
    mCE2 = np.zeros([len(Esize),3])
    
    strength_str = ['weakstr','weakdirect','weakweak']
    
    for n in range(len(Esize)):
        for k in range(3):
            NFILESTR = np.load(case_dir+'output/results'+str(Esize[n])+strength_str[k]+'.npz')
            Xassim = NFILESTR['arr_0'];Xspread = NFILESTR['arr_1']
            RMSEL2[n,k] = eval_by_msRMSE(Xassim[:,range(K)],XtrueL)
            RMSES2[n,k] = eval_by_msRMSE(Xassim[:,range(K,K*(J+1))],XtrueS)
            RMSSL2[n,k] = eval_by_msRMSS(Xspread[:,range(K)],XtrueL) 
            RMSSS2[n,k] = eval_by_msRMSS(Xspread[:,range(K,K*(J+1))],XtrueS)
            mCE2[n,k] = eval_by_CE(Xassim,Xtrue)
    np.savez(case_dir+'output/alter_loc.npz',RMSEL2,RMSES2,RMSSL2,RMSSS2,mCE2)
else:
    DATA_p = np.load(case_dir+'output/alter_loc.npz')
    RMSEL2=DATA_p['arr_0'];RMSES2=DATA_p['arr_1'];RMSSL2=DATA_p['arr_2'];
    RMSSS2=DATA_p['arr_3'];mCE2=DATA_p['arr_4']        

        
for k in range(3):
    RMSEL2[:,k]=np.sort(RMSEL2[:,k])[::-1]
    RMSES2[:,k]=np.sort(RMSES2[:,k])[::-1]
    RMSSL2[:,k]=np.sort(RMSSL2[:,k])[::-1]
    RMSSS2[:,k]=np.sort(RMSSS2[:,k])[::-1]
    mCE2[:,k]=np.sort(mCE2[:,k])
    mCE2[:,k]=np.sort(mCE2[:,k])        
        
    
        
# In[plt]
plt.figure(figsize=(16,8))      
plt.subplot(2,3,1)
plt.title('X-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSEL2[1::,0],width=0.28,label='XwZs',facecolor='green')
plt.bar(np.arange(5),RMSEL2[1::,1],width=0.28,label="XwZs2",facecolor='grey')
plt.bar(np.arange(5)+0.3,RMSEL2[1::,2],width=0.28,label='XwZw',facecolor='blue')
plt.ylim(0.02,0.25);
plt.text(-0.25,0.255,'(a)',fontsize=18)
plt.yticks([0.05,0.10,0.15,0.20],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSE',fontsize=18)
plt.grid(axis='y')
#plt.legend(fontsize=15)
plt.tight_layout(h_pad=4,w_pad=8)

plt.subplot(2,3,2)
plt.title('X-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSSL2[1::,0],width=0.28,label='XwZs',facecolor='green')
plt.bar(np.arange(5),RMSSL2[1::,1],width=0.28,label="XwZs2",facecolor='grey')
plt.bar(np.arange(5)+0.3,RMSSL2[1::,2],width=0.28,label="XwZw",facecolor='blue')
plt.ylim(0.02,0.25);
plt.text(-0.25,0.255,'(b)',fontsize=18)
plt.yticks([0.05,0.10,0.15,0.20],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSS',fontsize=18)
plt.grid(axis='y')
#plt.legend(fontsize=15)
plt.tight_layout(h_pad=4,w_pad=8)

plt.subplot(2,3,4)
plt.title('Z-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSES2[1::,0],width=0.28,label='XwZs',facecolor='green')
plt.bar(np.arange(5),RMSES2[1::,1],width=0.28,label="XwZs2",facecolor='grey')
plt.bar(np.arange(5)+0.3,RMSES2[1::,2],width=0.28,label="XwZw",facecolor='blue')
plt.ylim(0.3,0.65);
plt.text(-0.25,0.66,'(c)',fontsize=18)
plt.yticks([0.35,0.4,0.45,0.5,0.55],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSE',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15)

plt.subplot(2,3,5)
plt.title('Z-model',fontsize=18)
plt.bar(np.arange(5)-0.3,RMSSS2[1::,0],width=0.28,label='XwZs',facecolor='green')
plt.bar(np.arange(5),RMSSS2[1::,1],width=0.28,label="XwZs2",facecolor='grey')
plt.bar(np.arange(5)+0.3,RMSSS2[1::,2],width=0.28,label="XwZw",facecolor='blue')
plt.ylim(0.3,0.65);
plt.text(-0.25,0.66,'(d)',fontsize=18)
plt.yticks([0.35,0.4,0.45,0.5,0.55],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
#plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('MS-RMSS',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15)
plt.subplot(1,3,3)
plt.title('Coupled System',fontsize=18)
plt.bar(np.arange(5)-0.3,mCE2[1::,0],width=0.28,label='XwZs',facecolor='green')
plt.bar(np.arange(5),mCE2[1::,1],width=0.28,label="XwZs2",facecolor='grey')
plt.bar(np.arange(5)+0.3,mCE2[1::,2],width=0.28,label="XwZw",facecolor='blue')
plt.ylim(0.94,0.985);
plt.text(-0.25,0.9855,'(e)',fontsize=18)
plt.yticks([0.95,0.96,0.97,0.98],fontsize=15)
plt.xticks(np.arange(5),['20','40','80','160','320'],fontsize=15)
plt.xlabel('Ensemble size',fontsize=18)
plt.ylabel('Mean CE',fontsize=18)
plt.grid(axis='y')
plt.legend(fontsize=15,loc=2)