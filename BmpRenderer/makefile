# DETECTED_OS
ifeq ($(OS),Windows_NT)
	DETECTED_OS = Windows
else
	DETECTED_OS = $(shell sh -c 'uname -s 2>/dev/null || echo not')
endif

# LINK_TARGET & MK_LIB_DIR
ifeq ($(DETECTED_OS),Windows)
	LINK_TARGET = bmprenderer.lib
	MK_LIB_DIR  = if exist "Lib" {} else { mkdir Lib}
endif
ifeq ($(DETECTED_OS),Linux)
	LINK_TARGET = bmprenderer.a
	MK_LIB_DIR  = mkdir -p Lib
endif

# OBJS
OBJS = \
	bmpread.o \
	Bitmap.o \
	Window.o \
	Renderer.o

# REBUILDABLES
REBUILDABLES = $(LINK_TARGET) $(OBJS)

#
# Build commands
#
clean :
	rm -f $(REBUILDABLES)
	echo Clean done

all : $(LINK_TARGET)
	rm -f $(OBJS)

$(LINK_TARGET) : $(OBJS)
	$(MK_LIB_DIR)
	ar rcs ./Lib/$(LINK_TARGET) $^

#
# .o files
#
Bitmap.o :
	g++ -c -o $@ ./src/Bitmap.cpp

Window.o :
	g++ -c -o $@ ./src/API/Window.cpp

Renderer.o :
	g++ -c -o $@ ./src/Renderer.cpp

bmpread.o :
	g++ -c -o $@ ./Dependencies/libbmpread/bmpread.c
