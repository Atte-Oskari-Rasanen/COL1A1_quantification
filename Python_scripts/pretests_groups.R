library("dplyr")
library("ggpubr")
library("onewaytests")
df1 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_data.csv")
df1$Animal_ID <- as.character(as.character(df1$Animal_ID)) 

df_area <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Acoloc_Ahunu.csv") 
df_coloc <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Ncoloc.csv") 
df_tot <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Nhunu.csv") 
df_A_coloc_N_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Acoloc_Nhunu.csv") 
df_N_coloc_A_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Ncoloc_Ahunu.csv") 
df_N_coloc_N_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Ncoloc_Nhunu.csv") 

df_area[df_area == 0] <- NA
df_coloc[df_coloc == 0] <- NA
df_tot[df_tot == 0] <- NA
df_A_coloc_N_hunu[df_A_coloc_N_hunu==0]<-NA
df_N_coloc_A_hunu[df_N_coloc_A_hunu==0]<-NA
df_N_coloc_N_hunu[df_N_coloc_N_hunu==0]<-NA

df_area_all <- data.frame(stats = c(t(df_area)), stringsAsFactors=FALSE)
df_coloc_all <- data.frame(stats = c(t(df_coloc)), stringsAsFactors=FALSE)
df_total_all <- data.frame(stats = c(t(df_tot)), stringsAsFactors=FALSE)
df_A_coloc_N_hunu_all <- data.frame(stats = c(t(df_A_coloc_N_hunu)), stringsAsFactors=FALSE)
df_N_coloc_A_hunu_all <- data.frame(stats = c(t(df_N_coloc_A_hunu)), stringsAsFactors=FALSE)
df_N_coloc_N_hunu_all <- data.frame(stats = c(t(df_N_coloc_N_hunu)), stringsAsFactors=FALSE)

df_summary <- data.frame(matrix(ncol=2,nrow=6, dimnames=list(NULL, c("shapiro", "brown"))))

#shap_df_Ncoloc_Nhunu


#test for normality             not normal nor eq variance
shap_df_area <- sapply(df_area_all, shapiro.test)
shap_df_area
#test for variance
area_bs = bf.test(A_COL1A1_HUNU_AHUNU~Animal_ID, data = df1)
area_bs$p.value

#test for normality           normal and eq. variance
shap_df_coloc <- sapply(df_coloc_all, shapiro.test)
shap_df_coloc
#test for variance
coloc_bs = bf.test(N_COL1A1_HUNU~Animal_ID, data = df1)
coloc_bs$p.value
#################

#test for normality   normal and eq. variance
shap_df_total <- sapply(df_total_all, shapiro.test)
shap_df_total
#test for variance
total_bs = bf.test(N_HUNU~Animal_ID, data = df1)
total_bs$p.value
#################

#test for normality   #normal and eq. variance
shap_df_Acoloc_Nhunu <- sapply(df_A_coloc_N_hunu_all, shapiro.test)
shap_df_Acoloc_Nhunu
#test for variance
Acoloc_Nhunu_bs = bf.test(A_COL1A1_HUNU_N_HUNU~Animal_ID, data = df1)
Acoloc_Nhunu_bs$p.value
#################



#test for normality    does not fulfill normality but does variance and good freq
shap_df_Ncoloc_Ahunu <- sapply(df_N_coloc_A_hunu_all, shapiro.test)
shap_df_Ncoloc_Ahunu
#test for variance
Ncoloc_Ahunu_bs = bf.test(N_COL1A1_HuNu_AHunu~Animal_ID, data = df1)
Ncoloc_Ahunu_bs$p.value
##################

#normal and eq. variance
shap_df_Ncoloc_Nhunu <- sapply(df_N_coloc_N_hunu_all, shapiro.test)
shap_df_Ncoloc_Nhunu
shap_df_Ncoloc_Nhunu_bs = bf.test(N_COL1A1_HuNu_N_Hunu~Animal_ID, data = df1)

df1 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_data.csv")
df1$Animal_ID <- as.character(as.character(df1$Animal_ID)) 

df_area <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Acoloc_Ahunu.csv") 
df_coloc <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc.csv") 
df_tot <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Nhunu.csv") 
df_A_coloc_N_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Acoloc_Nhunu.csv") 
df_N_coloc_A_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc_Ahunu.csv") 
df_N_coloc_N_hunu <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc_Nhunu.csv") 

df_area[df_area == 0] <- NA
df_coloc[df_coloc == 0] <- NA
df_tot[df_tot == 0] <- NA
df_A_coloc_N_hunu[df_A_coloc_N_hunu==0]<-NA
df_N_coloc_A_hunu[df_N_coloc_A_hunu==0]<-NA
df_N_coloc_N_hunu[df_N_coloc_N_hunu==0]<-NA

df_area_all <- data.frame(stats = c(t(df_area)), stringsAsFactors=FALSE)
df_coloc_all <- data.frame(stats = c(t(df_coloc)), stringsAsFactors=FALSE)
df_total_all <- data.frame(stats = c(t(df_tot)), stringsAsFactors=FALSE)
df_A_coloc_N_hunu_all <- data.frame(stats = c(t(df_A_coloc_N_hunu)), stringsAsFactors=FALSE)
df_N_coloc_A_hunu_all <- data.frame(stats = c(t(df_N_coloc_A_hunu)), stringsAsFactors=FALSE)
df_N_coloc_N_hunu_all <- data.frame(stats = c(t(df_N_coloc_N_hunu)), stringsAsFactors=FALSE)

df_summary <- data.frame(matrix(ncol=2,nrow=6, dimnames=list(NULL, c("shapiro", "brown"))))


#test for normality                   normally distr., equal variance
shap_df_area <- sapply(df_area_all, shapiro.test)
shap_df_area
#test for variance
area_bs = bf.test(A_COL1A1_HUNU_AHUNU~Animal_ID, data = df1)
area_bs
area_bs$p.value

#test for normality             normally distributed and equal variance
shap_df_coloc <- sapply(df_coloc_all, shapiro.test)
shap_df_coloc
#test for variance
coloc_bs = bf.test(N_COL1A1_HUNU~Animal_ID, data = df1)
coloc_bs
coloc_bs$p.value
#################

#test for normality            # normally distrubuted and equal variance
shap_df_total <- sapply(df_total_all, shapiro.test)
shap_df_total
#test for variance
total_bs = bf.test(N_HUNU~Animal_ID, data = df1)
total_bs$p.value
#################

#test for normality   #normally distrb and equal variance
shap_df_Acoloc_Nhunu <- sapply(df_A_coloc_N_hunu_all, shapiro.test) 
shap_df_Acoloc_Nhunu
#test for variance
Acoloc_Nhunu_bs = bf.test(A_COL1A1_HUNU_N_HUNU~Animal_ID, data = df1)
Acoloc_Nhunu_bs$p.value
#################



#test for normality    normally distr and unequal variance
shap_df_Ncoloc_Ahunu <- sapply(df_N_coloc_A_hunu_all, shapiro.test)
shap_df_Ncoloc_Ahunu
#test for variance
Acoloc_Nhunu_bs = bf.test(N_COL1A1_HuNu_AHunu~Animal_ID, data = df1)
Acoloc_Nhunu_bs$p.value
##################
#test for normality    normally distr and equal variance
shap_df_Ncoloc_Nhunu <- sapply(df_N_coloc_N_hunu_all, shapiro.test)
shap_df_Ncoloc_Ahunu
#test for variance
Acoloc_Nhunu_bs = bf.test(N_COL1A1_HuNu_N_Hunu~Animal_ID, data = df1)
Acoloc_Nhunu_bs$p.value

