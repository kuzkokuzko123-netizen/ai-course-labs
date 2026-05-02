# -*- coding: utf-8 -*-
"""
Проверка установки snntorch и зависимости
Лабораторная работа №7
"""

import sys

def check_installation():
    """Проверка всех зависимостей."""
    print("=" * 60)
    print("ПРОВЕРКА УСТАНОВКИ ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    checks = {
        "Python": lambda: sys.version.split()[0],
        "PyTorch": lambda: __import__('torch').__version__,
        "snntorch": lambda: __import__('snntorch').__version__,
        "NumPy": lambda: __import__('numpy').__version__,
        "Matplotlib": lambda: __import__('matplotlib').__version__,
    }

    all_ok = True
    for name, check_fn in checks.items():
        try:
            version = check_fn()
            print(f"☑ {name}: {version}")
        except ImportError:
            print(f"☑ {name}: НЕ УСТАНОВЛЕН")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("☑ Все зависимости установлены корректно")
        return True
    else:
        print("☑ Некоторые зависимости отсутствуют")
        print("Выполните: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    check_installation()