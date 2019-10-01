Search.setIndex({docnames:["evaluation_inference","index","setup_dependencies","training","updown/config","updown/data","updown/data.datasets","updown/data.readers","updown/models","updown/models.updown_captioner","updown/modules","updown/modules.attention","updown/modules.cbs","updown/modules.updown_cell","updown/utils","updown/utils.checkpointing","updown/utils.common","updown/utils.constraints","updown/utils.decoding","updown/utils.evalai"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["evaluation_inference.rst","index.rst","setup_dependencies.rst","training.rst","updown/config.rst","updown/data.rst","updown/data.datasets.rst","updown/data.readers.rst","updown/models.rst","updown/models.updown_captioner.rst","updown/modules.rst","updown/modules.attention.rst","updown/modules.cbs.rst","updown/modules.updown_cell.rst","updown/utils.rst","updown/utils.checkpointing.rst","updown/utils.common.rst","updown/utils.constraints.rst","updown/utils.decoding.rst","updown/utils.evalai.rst"],objects:{"updown.config":{Config:[4,1,1,""]},"updown.config.Config":{_validate:[4,2,1,""],dump:[4,2,1,""]},"updown.data":{datasets:[6,0,0,"-"],readers:[7,0,0,"-"]},"updown.data.datasets":{EvaluationDataset:[6,1,1,""],EvaluationDatasetWithConstraints:[6,1,1,""],TrainingDataset:[6,1,1,""]},"updown.data.datasets.EvaluationDataset":{from_config:[6,2,1,""]},"updown.data.datasets.EvaluationDatasetWithConstraints":{from_config:[6,2,1,""]},"updown.data.datasets.TrainingDataset":{from_config:[6,2,1,""]},"updown.data.readers":{CocoCaptionsReader:[7,1,1,""],ConstraintBoxesReader:[7,1,1,""],ImageFeaturesReader:[7,1,1,""]},"updown.models":{updown_captioner:[9,0,0,"-"]},"updown.models.updown_captioner":{UpDownCaptioner:[9,1,1,""]},"updown.models.updown_captioner.UpDownCaptioner":{_decode_step:[9,2,1,""],_get_loss:[9,2,1,""],_initialize_glove:[9,2,1,""],forward:[9,2,1,""],from_config:[9,2,1,""]},"updown.modules":{attention:[11,0,0,"-"],cbs:[12,0,0,"-"],updown_cell:[13,0,0,"-"]},"updown.modules.attention":{BottomUpTopDownAttention:[11,1,1,""]},"updown.modules.attention.BottomUpTopDownAttention":{_project_image_features:[11,3,1,""],forward:[11,2,1,""]},"updown.modules.cbs":{ConstrainedBeamSearch:[12,1,1,""]},"updown.modules.cbs.ConstrainedBeamSearch":{search:[12,2,1,""]},"updown.modules.updown_cell":{UpDownCell:[13,1,1,""]},"updown.modules.updown_cell.UpDownCell":{_average_image_features:[13,3,1,""],forward:[13,2,1,""]},"updown.utils":{checkpointing:[15,0,0,"-"],common:[16,0,0,"-"],constraints:[17,0,0,"-"],decoding:[18,0,0,"-"],evalai:[19,0,0,"-"]},"updown.utils.checkpointing":{CheckpointManager:[15,1,1,""]},"updown.utils.checkpointing.CheckpointManager":{step:[15,2,1,""]},"updown.utils.common":{cycle:[16,4,1,""]},"updown.utils.constraints":{ConstraintFilter:[17,1,1,""],FiniteStateMachineBuilder:[17,1,1,""],add_constraint_words_to_vocabulary:[17,4,1,""]},"updown.utils.constraints.FiniteStateMachineBuilder":{_add_nth_constraint:[17,2,1,""],_connect:[17,2,1,""],build:[17,2,1,""]},"updown.utils.decoding":{select_best_beam:[18,4,1,""],select_best_beam_with_constraints:[18,4,1,""]},"updown.utils.evalai":{NocapsEvaluator:[19,1,1,""]},"updown.utils.evalai.NocapsEvaluator":{evaluate:[19,2,1,""]},updown:{config:[4,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"case":[7,13,17],"class":[2,4,6,7,9,11,12,13,15,17,19],"default":[3,4,6,9,12,13,15,17,18,19],"final":[2,17],"float":[6,15,17,19],"function":[12,18],"import":[2,4],"int":[6,9,11,12,13,15,17,18,19],"long":2,"return":[6,7,9,11,12,13,17,18,19],"static":[],"true":[0,6,7],"while":[6,13,17],CBS:[2,4,6,9,17],For:[3,4,7,9,11,13,15,17],GBs:7,IDs:3,NMS:[4,6,17],One:[15,19],That:17,The:[2,3,7,9,12,13,17,18],Then:17,These:[2,7,9,12,17,18],Use:6,Used:11,Using:16,__________:4,___________:13,__________________:13,__getitem__:7,__len__:7,_add_nth_constraint:17,_almost_:7,_average_image_featur:13,_connect:17,_decode_step:9,_get_loss:9,_initialize_glov:9,_num_main_st:17,_project_image_featur:11,_total_st:17,_valid:4,about:17,abs:[],accept:12,access:[4,6],accord:[17,18],accordingli:18,account:2,across:9,action:12,activ:2,actual:[11,13,19],adapt:[2,11,12,13],add:[0,17],add_constraint_words_to_vocabulari:17,added:[1,17],adjac:[6,9,12,17],after:[4,17],agraw:1,all:[3,4,6,9,17,19],allennlp:[4,6,9,12,17],allow:[2,4,19],along:6,alreadi:[],also:[3,9,12,18],alwai:17,amazonaw:2,among:18,anaconda:2,analog:[9,13],anderson2017up:1,anderson:[1,2,9,11,13],ani:[4,7,12,17],anim:[4,6,17],annot:[4,6,7,19],annotations_trainval2017:2,answer:1,anyth:7,anywher:2,api:[11,13,15],appear:9,appendix:17,appli:[11,12],apt:2,aption:1,architectur:[3,4,13],argpars:3,argument:[3,12],arrang:2,arrow:17,articl:[],arxiv:[],assum:4,attend:11,attent:[1,4,9,10,13],attention_projection_s:[4,9,13],attr:7,attribut:4,auth:2,author:1,automat:17,avail:[2,4,17],averag:13,avoid:[4,17,19],back:17,base:[1,2,3,4,6,7,9,11,12,13,15,17,19],baselin:[2,3,4,7],basi:17,basic:[9,13,18],batch:[4,6,9,11,13,16],batch_siz:[3,4,9,11,12,13,18],batra:1,beam:[1,3,4,6,9,12,17,18],beam_log_prob:18,beam_search:9,beam_siz:[4,9,12,18],beamsearch:[9,12,18],becaus:[6,7,9,11,13,16],befor:[4,9,11,13],behav:13,behind:2,being:[12,17],benchmark:1,best:[3,9,12,15,18,19],better:12,between:17,bewar:7,binari:[13,17],bject:1,blacklist:17,bleu:19,booktitl:1,bool:[6,7,9],both:[11,17,19],bottom:[1,2,4,9,11,13],bottomuptopdownattent:11,bound:[2,4,6,7,17],boundari:[9,12],box:[2,4,6,7,9,11,13,17],boxes_jsonpath:[6,7],browser:3,buehler:1,build:[17,18],build_vocabulari:2,built:6,burn:4,butd:13,cach:[11,13],cale:1,call:[2,9,11,13,15,17],callabl:12,callind:[],can:[2,3,4,6,7,12,17,19],candid:12,caption:[2,4,6,7,9,13,17,19],caption_token:9,captions_jsonpath:[6,7],captions_train2017:[2,4],captions_val2017:2,captur:9,cardin:17,care:13,categori:[7,17],caus:16,cbs:[1,2,4,9,10,17],cbs_select_best_beam:[],cell:[4,9,13],certain:17,chang:18,check:12,checkpoint:[0,1,14,19],checkpointmanag:[3,15],chen:1,chosen:[],chri:1,cider:[3,19],cite:1,ckpt:15,ckpt_manag:15,class_hierarchi:[2,4],classmethod:[6,9],clean:[11,13],cli:2,clip:4,clip_gradi:4,clone:2,close:[2,12,15],cnn:7,cnstrain:3,coco:[1,2,4,6,7,17,19],coco_train2017:7,coco_train2017_resnet101_faster_rcnn_genome_adapt:[],coco_train2017_vg_detector_features_adapt:[2,4],coco_val2017:7,coco_val2017_vg_detector_features_adapt:2,cococaptionsread:7,cocodataset:2,code:1,codebas:1,collate_fn:6,collect:[4,17],column:4,com:2,comma:[4,6,17],common:[1,11,14],commonli:17,comput:[1,4,9,11,12,13,17],conda:2,condit:12,confer:1,confid:[2,4,6,17],config:[0,1,3,6,9],config_overrid:4,config_yaml:4,configur:[3,4],conflict:4,connect:17,consecut:17,consid:[1,12,17],constrain:[1,3,4,9,12,17],constrainedbeamsearch:[6,9,12,18],constraint:[1,2,4,6,9,12,14,18],constraint_wordform:[2,4],constraintboxesread:7,constraintfilt:17,construct:17,constructor:16,contain:[3,4,6,7,9,11,12,13,17,18],control:4,coordin:7,copi:3,correspond:[4,6,7,9,17,18],could:[2,6,17],cover:17,cpu:3,creat:2,cross:9,current:[12,13],curv:3,cvpr:1,cycl:16,damien:1,data:[1,2,4,9,16,17],dataload:[6,16],dataparallel:15,dataset:[1,5,7],decai:4,declar:[2,3],decod:[1,2,3,4,6,9,11,12,13,14,17],defin:[3,17,18],definit:[4,18],desai:1,descend:18,destin:17,detail:[2,6,17],detect:[2,4,6,7,17],detector:[2,7],dev:2,develop:2,devi:1,devic:16,dfkjgmlk:[],dhruv:1,dict:[4,9,12,13,15,16,19],differ:[3,9,11,13],dimens:[3,11,12,17,18],dir:3,directli:[0,6,9],directori:[2,4,15],discard:12,disk:7,distribut:2,divers:12,docstr:12,doe:[7,9],dog:[4,6,17],doing:19,domain:[9,19],done:[16,19],down:[1,4,9,11,13],drop:18,dump:4,duplic:17,dure:[0,3,4,6,9,15,17,19],dynam:17,each:[7,9,11,12,13,17],earli:19,effici:4,eight:17,either:[2,4,9,17,18],element:[9,11,12],els:17,elsewher:4,embed:[0,3,4,7,9,13],embedding_s:[0,4,9,13],empti:15,enabl:19,end:4,end_index:12,entir:19,entropi:9,environ:2,epoch:[15,16],epoch_or_iter:15,equal:17,error:4,etc:[4,6,17,19],evalai:[0,1,2,14],evalu:[1,3,4,9,11,13,19],evaluationdataset:6,evaluationdatasetwithconstraint:6,even:2,everi:3,exampl:[3,4,7,15,17,19],except:12,execut:[3,11,13],exhaust:19,exist:15,expand:17,expect:12,experi:3,explod:4,extra:17,extract:[2,4,6,7],extractor:2,fail:2,fals:[4,7,9],far:17,faster:2,featur:[4,6,7,9,11,13],feature_s:7,features_h5path:7,few:[3,19],field:[6,7,17],fig:13,figur:17,file:[0,3,4,6,7,15,17],file_path:4,filename_prefix:15,filesystem:2,filter:[4,6,17],find:[1,4,9,12],finit:[6,9,12,17],finitestatemachinebuild:17,fire:17,first:[3,4,6,9,11,12,13,17],fix:[16,17],flag:0,follow:[0,2,3,7,13,15],forc:9,form:[4,6,17],format:[2,4,6,7,19],forward:[9,11,13],four:7,freitag:12,from:[2,3,4,6,7,9,11,12,13,16,17],from_config:[6,9],from_stat:17,frozen:[0,3,9],fsm:[6,9,12,17,18],fulli:[3,17],gener:[0,4,6,16,17],genom:2,get:[0,1,2],git:2,github:2,give:12,given:[9,12,13,17,18],given_constraint:18,glove:[0,3,9],goe:[17,18],gould:1,gpu:0,gradient:4,ground:[6,9],group_siz:12,hand:18,happen:[4,12],harm:16,harsh:1,has:[12,18],have:[2,7,12,16,17],height:7,held:[2,19],help:9,helper:17,henc:[9,17],here:[4,6,7,9],hidden:[4,9,13],hidden_s:[4,9,13],hidden_st:13,hierarchi:[2,4,6,17],hierarchy_jsonpath:[6,17],higher:[2,4,6,15,17],highest:18,hit:15,how:[1,18,19],howev:19,http:2,human:[2,17],hydrant:17,hyper:4,hyperparamet:[3,4],iccv:1,ids:[0,3,7],ignor:[9,17],imag:[1,4,6,7,9,11,13,17],image_bottomup_featur:7,image_featur:[9,11,13],image_feature_s:[4,9,11,13],image_features_h5path:6,image_features_mask:11,image_id:[7,19],imagefeaturesread:7,immedi:17,implement:[1,7,12,18],in_memori:[6,7],incomplet:17,increas:17,index:[6,7,9,12,17],indic:[12,17],inf:12,infer:[1,6,9],infer_box:4,infer_capt:4,infer_featur:4,infinit:19,info:[2,4],initi:[3,4,9,12,13,17],inproceed:1,input:[4,9,13],instanc:[6,7,9,11,13],instanti:[4,6,9],instead:15,instruct:7,integ:7,intern:[1,9],introduc:12,iou:[4,6,17],iter:[3,4,16,19],itertool:16,its:17,itself:[11,13],jain:1,johnson:1,journal:[],json:[0,2,4,6,7,17],just:12,karan:1,keep:[3,11,12,13,15,17],kei:[3,4,7,9,13,19],kwarg:[6,9],languag:[4,6,9,13],last:[11,12,13],later:17,ldgkb:[],leak:16,learn:[4,15,19],least:[7,11,13,17,18,19],lee:1,lei:1,length:[4,6,7,9,12],less:[17,18,19],lesser:[9,11,13],let:4,libxml2:2,libxstl1:2,like:[4,9,12,17],likelihood:[9,18],linear:15,linearli:4,list:[4,6,17,19],load:[6,7],local:19,localhost:3,log:[9,12],log_prob:12,logdir:3,logit:9,longer:[4,6,9],loop:[17,19],loss:[3,9],lower:15,lru:[11,13],lstm:[4,9,11,13],lstmcell:[9,13],machin:[6,9,12,17],made:2,mai:[4,7,9,12,16,17],main:[12,17],maintain:[11,13],make:[6,17],mammal:17,manag:[3,4],mani:18,map:[6,9,17],mark:1,mask:[9,11,12,13],matric:6,matrix:[9,12,17],max:15,max_caption_length:[4,6,9],max_decoding_step:18,max_given_constraint:[4,6,17],max_step:12,max_words_per_constraint:[4,17],maxim:9,maximum:[4,6,9,11,12,13,17],mean:[12,13,17],meaning:9,member:17,memori:[6,7,16],meta:2,meteor:19,method:[7,12,17],metric:[3,15,19],might:[9,11,13],min:15,min_constraints_to_satisfi:[4,9,18],miniconda:2,minimum:[4,9,18],mode:15,model:[0,1,4,6,15,19],modif:4,modul:[1,4,9,15],momentum:4,more:[2,3,6,9,12,17],most:[4,9,12],much:7,multi:[4,17],multipl:[9,17],must:[4,7],name:[4,6,7,17],namespac:17,need:[15,17],nest:[4,19],net_beam_s:9,neural:12,never:9,next:[12,13,17],nms_threshold:[4,6,17],nocap:[0,2,3,4,6,7,19],nocaps2019:1,nocaps_test:7,nocaps_test_image_info:2,nocaps_test_oi_detector_box:2,nocaps_test_resnet101_faster_rcnn_genome_adapt:[],nocaps_test_vg_detector_features_adapt:2,nocaps_v:7,nocaps_val_image_info:[2,4],nocaps_val_oi_detector_box:[2,4],nocaps_val_resnet101_faster_rcnn_genome_adapt:[],nocaps_val_vg_detector_features_adapt:[2,4],nocapsevalu:19,node:12,non:[12,15],none:[9,11,12,13,17,19],nonetyp:[9,11,13,16,19],note:[4,15,17,19],notion:16,now:2,num_box:[7,9,11,13],num_constraint:9,num_epoch:15,num_fsm_stat:12,num_imag:7,num_iter:4,num_main_st:17,num_stat:[9,18],num_total_st:17,number:[3,4,6,9,12,16,17,18,19],numpi:4,object:[2,4,6,7,12,15,17,19],observ:15,occurr:6,onaizan:12,onc:[11,13],one:[2,3,7,9],onli:[0,1,3,7,17,18],open:[2,4,6,7,17],optim:[3,4,15],option:[4,6,7,9,12,13,15,17,18,19],order:17,org:[1,2],origin:[3,17],other:[3,7,12,17],otherwis:17,our:[2,3,4,7,17],out:[9,18,19],output:[0,11,13],ovel:1,over:[9,11],overal:3,overlap:17,overrid:[0,3,4],overriden:4,own:3,packag:[2,4],pad:[9,11,13],pair:[3,9],paper:[1,3,4,17],paramet:[4,6,7,9,11,12,13,15,17,18,19],parikh:1,part:17,particular:[3,4,9,13,17],pass:[3,7,12],path:[0,3,4,6,7,15,17],pepper:17,per:[2,6,9,12,17],per_node_beam_s:12,perform:[3,4,9,13,15,17,19],period:15,perpetu:16,peter:1,phase:[4,19],pip:2,pleas:1,plural:[2,4,6,17],point:17,pool:13,port:3,posit:17,possibl:[3,4,6,12,17],potenti:12,practic:19,pre:[2,4,6,7],predict:[0,4,6,9,12,13,17,19],prefix:15,prepar:17,preprint:[],pretrain:[0,2],previou:9,previous_predict:9,primari:7,privat:[2,19],probabl:[9,12],process:7,produc:7,profil:2,project:[4,9,11,13],project_root:2,projection_s:11,propos:[1,2,17],provid:[2,4,6,9,11,13,17],pth:[0,15],publicli:4,python3:2,python:[0,2,3],pytorch:[4,6,7,11,15],queri:11,query_s:11,query_vector:11,question:1,rais:4,ram:7,random:[4,16],random_se:4,randomli:[3,4,9,17],rang:15,rare:17,rate:[4,15,19],rcnn:2,reach:17,read:7,readabl:[2,4],reader:[1,5],recommend:[2,4,7,9,19],record:15,recurr:9,refer:[3,6],region:2,regular:9,rel:4,relev:4,remain:[11,13],remot:19,remov:17,repo:2,repositori:2,repres:[6,9,12,17],represent:17,reproduc:[3,4],requir:[2,4,6,7,19],reset:17,reset_st:17,resolv:17,respons:12,result:[0,3,12],retriev:19,rishabh:1,root:4,roug:19,row:3,run:[3,9],said:17,salt:17,same:[9,11,13],sampl:4,satisfi:[2,4,6,9,12,17,18],save:[0,4,11,13,15,17],schedul:[15,19],score:[3,4,6,17],script:[0,2,3,19],search:[1,3,4,9,12,17],second:[3,4,6,12,17],see:12,seed:4,seen:[11,13],select:[2,4,6,9,17,18,19],select_best_beam:18,select_best_beam_with_constraint:[6,18],self:[4,9,11,12,13,15,17,19],semant:9,sensibl:[4,17],separ:[4,6,17],sequenc:[4,6,9,12,18],sequenti:4,serial:[3,15],serialization_dir:[3,15],serv:[17,19],set:[3,7,12,17,18],set_token:2,setup:1,sever:17,sfklkv:[],sgd:[4,15],shape:[7,9,11,12,13,17,18],should:[3,7,12],show:[9,19],shown:17,signatur:[11,12,13],similar:[7,13],simpli:7,singl:[11,13],singular:[2,4,6,17],site:2,size:[3,4,7,9,11,12,13],smaller:12,softmax:12,some:[11,12,13,17],sometim:7,sort:18,sourc:[1,4,6,7,9,11,12,13,15,16,17,18,19],sown:[],specif:[3,6,12,17,19],specifi:[0,4,6,12,15,17,18],spice:19,split:[2,7,19],start:[12,17,18],start_predict:12,start_stat:12,state:[4,6,9,12,13,15,17,18],state_dict:15,stefan:1,step:[9,11,12,13,15],stepfunctiontyp:12,stephen:1,stop:19,stove:4,str:[4,6,7,9,12,13,15,16,17,19],strategi:12,structur:[2,4,19],style:3,sub:17,submiss:19,submit:[0,19],substate_idx:17,suffici:7,sum:9,support:[1,3,17,18],suppress:[4,6,17],symlink:[2,4],synonym:2,tabl:3,take:[9,12,13,17,19],target:[9,12],target_mask:9,teacher:9,ten:7,tenei:1,tensor:[6,9,11,12,13,15,16,17,18],tensorboard:3,term:17,test:[0,2,4,6,7,17,19],test_capt:[],test_featur:[],textual:[4,9,11,13],than:[4,6,7,9,12,17,18],thank:12,thei:[9,18,19],them:[2,12,17,19],therefor:12,thi:[0,1,3,4,6,7,9,11,12,13,15,16,17,18,19],those:3,three:17,threshold:[2,4,6,17],through:[2,3,4,7,17],till:4,time:[4,9,11,12,13],titl:1,tmp:15,to_stat:17,togeth:19,token:[2,6,7,9,12,13,17],token_embed:13,too:[7,9,17],tool:19,top:[1,4,9,11,13,17],torch:[6,9,11,12,13,15,16,17,18],total:[9,18],track:[3,15,17],train2017:[2,4,6,7],train:[0,1,2,4,6,7,9,11,13,15,16,19],train_capt:4,train_featur:4,trainingdataset:6,transit:[6,9,12,17],translat:12,trim:17,truncat:[4,6,9],truth:[6,9],tsv:[2,4,6,17],tupl:[9,12,13,17],two:[4,6,7,12,17],txt:2,type:[15,19],typic:[7,11],under:[2,3,17],unexpec:16,union:[9,11,13,15,19],unit:13,unknown:9,unnorm:9,unreach:17,unus:17,updat:[9,12,13,15,17],updown:2,updown_caption:[1,8],updown_cel:[1,10],updown_nocaps_v:3,updown_plus_cb:3,updown_plus_cbs_nocaps_v:3,updowncaption:[6,9,13],updowncel:[9,11,13],use:[0,4,7,9,11,13,17],use_cb:[0,4,9],used:[3,6,7,9,11,12,16,17,19],useful:1,useless:17,user:19,uses:3,using:[0,1,2,3,7,9,11,12,19],usual:[6,7,12],util:[1,3,6,7],utter:17,val:[0,2,3,4,6,7,19],val_capt:[],val_featur:[],val_loss:15,valid:[0,4,15,17],valu:[3,4,11,12,13,15],variabl:13,vector:[9,11],veri:[13,19],version:2,via:2,vision:1,visit:3,visual:[1,2],vocab_s:[9,12,17],vocabulari:[4,6,9,12,17],wai:2,wang:1,want:12,weight:[4,9,11,13],weight_decai:4,well:[1,17],when:[6,9,12,15],where:[9,11,12,19],whether:[4,6,7,9],which:[2,3,4,6,11,12,13,15,16,17,18,19],wide:4,width:[7,12],wish:[0,2,9],without:19,wood:4,word:[2,3,4,6,7,9,13,17],wordform:[2,4],wordforms_tsvpath:[6,17],work:[17,19],worker:3,worri:17,would:[3,9,11,13,17],wrap:6,write:3,www:2,xiaodong:1,xinlei:1,yaml:[0,3,4],year:1,yet:9,yield:16,you:[0,1,4,7,12],your:[1,2,12],your_token_her:2,yufei:1,zero:[4,9,11,12,13,17],zhang:1,zip:2},titles:["How to evaluate or do inference?","UpDown Captioner Baseline for <code class=\"docutils literal notranslate\"><span class=\"pre\">nocaps</span></code>","How to setup this codebase?","How to train your captioner?","updown.config","updown.data","updown.data.datasets","updown.data.readers","updown.models","updown.models.updown_captioner","updown.modules","updown.modules.attention","updown.modules.cbs","updown.modules.updown_cell","updown.utils","updown.utils.checkpointing","updown.utils.common","updown.utils.constraints","updown.utils.decoding","updown.utils.evalai"],titleterms:{CBS:3,Using:0,addit:3,all:2,annot:2,api:1,attent:11,baselin:1,beam:[0,2],build:2,caption:[1,3],cbs:12,checkpoint:[3,15],codebas:2,common:16,config:4,constrain:[0,2],constraint:17,data:[5,6,7],dataset:6,decod:18,depend:2,detail:3,download:2,evalai:19,evalu:[0,2],featur:2,file:2,gpu:3,guid:1,how:[0,2,3],imag:2,infer:0,instal:2,log:3,model:[3,8,9],modul:[10,11,12,13],multi:3,nocap:1,option:2,reader:7,refer:1,save:3,search:[0,2],server:2,set:2,setup:2,thi:2,train:3,updown:[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],updown_caption:9,updown_cel:13,use:2,user:1,util:[14,15,16,17,18,19],vocabulari:2,without:3,you:2,your:3}})