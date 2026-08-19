"""Cac co (flag) tam thoi dung CHUNG giua online/app.py va online/submission_pipeline.py - tach
rieng file nay de 2 noi luon doc CUNG 1 gia tri (tranh sua 1 cho quen sua cho kia, lech hanh vi
giua KIS/Q&A)."""
from __future__ import annotations

# 2026-08-17 (GAC LAI de TEST, theo yeu cau nguoi dung sau bug that): LLM query_planner.plan_
# query() phan ra entity KHONG ON DINH giua cac lan goi CUNG 1 cau (khong sua duoc bang cache -
# de thi that moi cau chi hoi DUNG 1 LAN, hen xui van fail dung lan do) - co lan TU BIA THEM
# entity khong co trong cau goc (vd cau ve "chiếc xuồng gỗ..." khong nhac gi toi thung/vai, LLM
# van bia ra 1 entity resolve nham thanh nhan "Barrel"), bien thanh HARD FILTER SAI, bop hep sai
# corpus (vd con 4675/369589 frame khong lien quan) - lam sai toan bo ket qua ma nguoi dung
# khong he thay dau hieu gi bat thuong (query/UI y het lan dung).
#
# True = TAT hard-filter tu LLM entity (chi con giu filter tu khung Object VE TAY tren canvas -
# dang tin cay hon nhieu vi la y dinh THAT cua nguoi dung, khong phai LLM doan/bia). Doi ve
# False de bat lai NHU CU khi da co huong sua tan goc (vd tang do nghiem ngat SYSTEM_PROMPT cua
# query_planner.py, hoac co benchmark that de kiem chung truoc/sau) va muon so sanh lai.
DISABLE_LLM_ENTITY_HARD_FILTER = True
