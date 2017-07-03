#
#  makefile for PsiQuaSP library
#


all: psiquasp

include options.mk

psiquasp:
	@echo
	@echo Building PsiQuaSP library
	@echo
	@cd src && $(MAKE) all


