# ==== Configuración ====
CXX := g++
# DBG
# CXXFLAGS := -std=gnu++20 -O3 -ffast-math -fno-math-errno -ffp-contract=fast -fopenmp -Iinclude

# Perfo
CXXFLAGS := -std=gnu++20 \
            -O3 \
            -march=native -mtune=native \
            -ffast-math -fno-math-errno -ffp-contract=fast \
            -funroll-loops \
            -fopenmp \
            -DNDEBUG \
            -Iinclude

LDFLAGS :=
LDLIBS := -fopenmp

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := build/bin
BIN := $(BIN_DIR)/app

SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC))

# ==== Reglas ====
.PHONY: all clean run debug

all: $(BIN)

# Ejecutable final
$(BIN): $(OBJ) | $(BIN_DIR)
	@echo "Linking $@"
	$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

# Compilar cada .cpp -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Crear directorios si no existen
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Ejecutar
run: $(BIN)
	$(BIN)

# Compilar con flags de debug (-g -O0)
debug: CXXFLAGS += -g -O0
debug: clean all

# Limpiar
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
