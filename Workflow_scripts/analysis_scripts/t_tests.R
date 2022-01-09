library("dplyr")
library("ggpubr")
library("onewaytests")


df1 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_data.csv")
df1 <- read.csv("/home/atte/Documents/PD_images/datas_known_ids/group1/group1_data_orig.csv")

df1$Animal_ID <- as.factor(df1$Animal_ID)

means_df1 <- df1 %>%
  group_by(Animal_ID) %>%
  summarise(mean_area = mean(A_COL1A1_A_HUNU), mean_numb = mean(N_COL1A1_HuNu_N_Hunu), n = n())


df2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_data.csv")
df2 <- read.csv("/home/atte/Documents/PD_images/datas_known_ids/group2/group2_data_orig.csv")

df2$Animal_ID <- as.factor(df2$Animal_ID)

means_df2 <- df2 %>%
  group_by(Animal_ID) %>%
  summarise(mean_area = mean(A_COL1A1_A_HUNU), mean_numb = mean(N_COL1A1_HuNu_N_Hunu), n = n())

df1$N_COL1A1_HUNU
mean(df1[['N_COL1A1_HUNU']])
mean(df2[['N_COL1A1_HUNU']])

df_summary_boxplot <- data.frame(matrix(ncol=2,nrow=14, dimnames=list(NULL, c("Group 1", "Group 2"))))

#df_summary_boxplot[1, :]<-df1$N_COL1A1_HUNU
total <- paste(df1$N_COL1A1_HUNU,df2$N_COL1A1_HUNU)
total

df_summary <- data.frame(matrix(ncol=2,nrow=6, dimnames=list(NULL, c("t test", "Wilcoxon"))))
df_summary
rownames <- c("Acoloc/Ahunu", "Ncoloc", "Acoloc/Nhunu", "Ncoloc/Ahunu", "Nhunu", "Ncoloc/Nhunu")
row.names(df_summary) <- rownames
#Acoloc_Ahunu   



#mutate(norm_N_COL1A1_HUNU_N_HUNU = (df1$N_COL1A1_HUNU-df1$N_HUNU)/df1$N_HUNU)
norm_N_COL1A1_HUNU_N_HUNU_1 = (((df1$N_COL1A1_HUNU-(df1$N_HUNU/2))/df1$N_HUNU))
norm_N_COL1A1_HUNU_N_HUNU_2 = (((df2$N_COL1A1_HUNU-(df2$N_HUNU/2))/df2$N_HUNU))
ncoloc_n_hunu_norm <- t.test(norm_N_COL1A1_HUNU_N_HUNU_1, norm_N_COL1A1_HUNU_N_HUNU_2)
ncoloc_n_hunu_norm
acoloc_hunu_norm_bf <- wilcox.test(norm_N_COL1A1_HUNU_N_HUNU_1, norm_N_COL1A1_HUNU_N_HUNU_2)
acoloc_hunu_norm_bf

#calculate the means of each aninmal for the metric norm_N_COL1A1_HUNU_N_HUNU
mean_anim_id <-  c(rep('A7',3), rep('A2',4), rep('A20',4), rep('A14',3))
mean_anim_id2 <- c(rep('A25',4), rep('A18', 4), rep('A6',5), rep('A12', 5))
values <- c(norm_N_COL1A1_HUNU_N_HUNU_1, norm_N_COL1A1_HUNU_N_HUNU_2)
A7 <- c(head(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1,3), 0)
A2 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[4:7])
A20 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[8:11])
A14 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[11:14])

df_norm1 <- data.frame(A7,A2,A20,A14)

A25 <- c(head(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2,4), 0)
A18 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[5:8], 0)
A6 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[9:13])
A12 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[14:18])

df_norm2 <- data.frame(A25,A18,A6,A12)

g1_norm <- colMeans(df_norm1)
g2_norm <- colMeans(df_norm2)

values_mean_norm <- c(g1_norm, g2_norm)
groups_mean <- c(rep('1',4), rep('2',4))
df_norm_mean <- data.frame(groups_mean,values_mean_norm)

df_N_coloc_N_hunu <- t.test(g1,g2)
df_N_coloc_N_hunu

Acol1a1_Ahunu <- t.test(df1$A_COL1A1_A_HUNU, df2$A_COL1A1_A_HUNU)
Acol1a1_Ahunu
Acol1a1_Ahunu_bs <- wilcox.test(df1$A_COL1A1_A_HUNU, df2$A_COL1A1_A_HUNU)
Acol1a1_Ahunu_bs

#t tests means
mean_acol1a1_ahunu <- t.test(means_df1$mean_area, means_df2$mean_area)
mean_acol1a1_ahunu
mean_acol1a1_ahunu <- t.test(means_df1$mean_numb, means_df2$mean_numb)
mean_acol1a1_ahunu

Acoloc_Ahunu <- t.test(df1$A_COL1A1_HUNU_AHUNU, df2$A_COL1A1_HUNU_AHUNU)#  test for normality  fulfills normality principle but not variance
Acoloc_Ahunu
df_summary[1,1] <- Acoloc_Ahunu$p.value
Acoloc_Ahunu_bs <- wilcox.test(df1$A_COL1A1_HUNU_N_HUNU, df2$A_COL1A1_HUNU_N_HUNU)
Acoloc_Ahunu_bs
df_summary[1,2] <- Acoloc_Ahunu_bs$p.value
hist(df1$A_COL1A1_HUNU_AHUNU, 
     main="group 1 A_COL1A1_HuNu_A_Hunu", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)
hist(df2$A_COL1A1_HUNU_AHUNU, 
     main="group 2 A_COL1A1_HuNu_A_Hunu", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)




Ncoloc <- t.test(df1$N_COL1A1_HUNU, df2$N_COL1A1_HUNU)  #ormality not fulfilled but variance has been
Ncoloc
Ncoloc_bs <- wilcox.test(df1$N_COL1A1_HUNU, df2$N_COL1A1_HUNU)
Ncoloc_bs

library(ggpubr)
library(tidyverse)
library(rstatix)

library(ggplot2)
ggplot(N,aes(x=CONDITION,y=BMIS,col=CONDITION))+geom_boxplot()

# Create the plot
myplot <- ggboxplot(selfesteem, x = "time", y = "score", add = "point")
# Add statistical test p-values
stat.test <- stat.test %>% add_xy_position(x = "time")
myplot + stat_pvalue_manual(stat.test, label = "p.adj.signif")

df_summary[2,1] <- Ncoloc$p.value
df_summary[2,2] <- Ncoloc_bs$p.value
hist(df1$N_COL1A1_HuNu_N_Hunu, 
     main="group 1 N_COL1A1_HuNu_N_Hunu", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)
hist(df2$N_COL1A1_HuNu_N_Hunu, 
     main="group 2 N_COL1A1_HuNu_N_Hunu", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)
hist(df1$N_COL1A1_HUNU, 
     main="group 1 N_COL1A1_HUNU", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)


hist(df1$A_COL1A1_HUNU_N_HUNU, 
     main="group 1 A_COL1A1_HUNU_N_HUNU", 
     xlab="x", 
     border="light blue", 
     col="blue", 
     las=1, breaks=15
)



Acoloc_Nhunu <-  t.test(df1$A_COL1A1_HUNU_N_HUNU, df2$A_COL1A1_HUNU_N_HUNU)  #fulfills normality and variance but freq looks weird!
Acoloc_Nhunu
Acoloc_Nhunu_bs <- wilcox.test(df1$A_COL1A1_HUNU_N_HUNU, df2$A_COL1A1_HUNU_N_HUNU)
Acoloc_Nhunu_bs
df_summary[3,1] <- Acoloc_Nhunu$p.value
df_summary[3,2] <- Acoloc_Nhunu_bs$p.value



Ncoloc_Ahunu <- t.test(df1$N_COL1A1_HuNu_AHunu, df2$N_COL1A1_HuNu_AHunu) #not normal but good variance
Ncoloc_Ahunu
Ncoloc_Ahunu_bs <- wilcox.test(df1$N_COL1A1_HuNu_AHunu, df2$N_COL1A1_HuNu_AHunu)
Ncoloc_Ahunu_bs
df_summary[4,1] <- Ncoloc_Ahunu$p.value
df_summary[4,2] <- Ncoloc_Ahunu_bs$p.value


Nhunu <- t.test(df1$N_HUNU, df2$N_HUNU)
#Nhunu <- wilcox.test(df1$N_HUNU, df2$N_HUNU)  #shapiro pval: 0.003232114 b-s pval:0.0853691
Nhunu

df_summary[5,1] <- NA
df_summary[5,2] <- Nhunu$p.value


Ncoloc_Nhunu <- t.test(df1$N_COL1A1_HuNu_N_Hunu, df2$N_COL1A1_HuNu_N_Hunu) #fulfills all criteria
Ncoloc_Nhunu

Ncoloc_Nhunu_bs <- wilcox.test(df1$N_COL1A1_HuNu_N_Hunu, df2$N_COL1A1_HuNu_N_Hunu)
Ncoloc_Nhunu_bs
df_summary[6,1] <- Ncoloc_Nhunu$p.value
df_summary[6,2] <- NA


df_summary


