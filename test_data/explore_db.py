#!/usr/bin/env python3

import sqlite3
import csv
import sys
from pathlib import Path


TABLES = ["graphs", "nodes", "edges", "targets", "labels"]


def connect_db(db_path):
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    return sqlite3.connect(db_path)


def show_tables(conn):
    print("\n=== TABLES ===")

    cursor = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """)

    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        count = conn.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

        print(f"  {table:<15} {count:,} rows")


def show_schema(conn, table):
    if table not in TABLES:
        print(f"Unknown table: {table}")
        return

    print(f"\n=== SCHEMA: {table} ===")

    cursor = conn.execute(f'PRAGMA table_info("{table}")')

    rows = cursor.fetchall()

    if not rows:
        print("No columns found.")
        return

    print(f"{'Column':<25} {'Type':<15} {'Nullable':<10} {'Primary Key'}")
    print("-" * 70)

    for row in rows:
        # row = cid, name, type, notnull, default, pk
        cid, name, col_type, notnull, default, pk = row

        print(
            f"{name:<25} "
            f"{col_type:<15} "
            f"{'NO' if notnull else 'YES':<10} "
            f"{'YES' if pk else 'NO'}"
        )


def preview_table(conn, table, limit=10):
    if table not in TABLES:
        print(f"Unknown table: {table}")
        return

    print(f"\n=== {table.upper()} — first {limit} rows ===")

    cursor = conn.execute(
        f'SELECT * FROM "{table}" LIMIT ?',
        (limit,)
    )

    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    if not rows:
        print("(empty table)")
        return

    # Print column names
    print("\t".join(columns))
    print("-" * 100)

    for row in rows:
        print("\t".join(str(value) for value in row))


def search_database(conn, search_term):
    print(f"\n=== SEARCHING FOR: {search_term!r} ===")

    found = False

    for table in TABLES:
        # Check whether table exists
        exists = conn.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
        """, (table,)).fetchone()

        if not exists:
            continue

        columns = conn.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()

        text_columns = [
            column[1]
            for column in columns
            if column[2].upper() in (
                "TEXT",
                "VARCHAR",
                "CHAR",
                ""
            )
        ]

        for column in text_columns:
            query = f'''
                SELECT *
                FROM "{table}"
                WHERE "{column}" LIKE ?
                LIMIT 20
            '''

            rows = conn.execute(
                query,
                (f"%{search_term}%",)
            ).fetchall()

            if rows:
                found = True

                print(f"\n[{table}.{column}]")

                for row in rows:
                    print(row)

    if not found:
        print("No matches found.")


def run_query(conn, query):
    print("\n=== QUERY RESULT ===")

    try:
        cursor = conn.execute(query)

        # SELECT-like query
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            print("\t".join(columns))
            print("-" * 100)

            for row in rows:
                print("\t".join(str(value) for value in row))

            print(f"\n{len(rows):,} row(s)")
        else:
            conn.commit()
            print(f"Query executed. {cursor.rowcount} row(s) affected.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")


def export_query(conn, query, filename):
    try:
        cursor = conn.execute(query)

        if not cursor.description:
            print("This query does not return rows.")
            return

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        print(f"Exported {len(rows):,} rows to {filename}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")


def menu(conn):
    while True:
        print("""
========================================
 SQLite Database Explorer
========================================

1. List tables
2. Show table schema
3. Preview table
4. Search database
5. Run SQL query
6. Export SQL query to CSV
7. Exit
""")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            show_tables(conn)

        elif choice == "2":
            table = input(
                "Table (graphs/nodes/edges/targets/labels): "
            ).strip()

            show_schema(conn, table)

        elif choice == "3":
            table = input("Table: ").strip()

            try:
                limit = int(input("Number of rows [10]: ") or "10")
            except ValueError:
                limit = 10

            preview_table(conn, table, limit)

        elif choice == "4":
            term = input("Search for: ").strip()

            if term:
                search_database(conn, term)

        elif choice == "5":
            print("\nEnter SQL query.")
            print("Example: SELECT * FROM nodes LIMIT 20")
            print("Type your query and press Enter.\n")

            query = input("SQL> ").strip()

            if query:
                run_query(conn, query)

        elif choice == "6":
            query = input("SQL query: ").strip()
            filename = input("Output CSV filename: ").strip()

            if query and filename:
                export_query(conn, query, filename)

        elif choice == "7":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")


def main():

    db_path = "test_NHCs.db"

    conn = connect_db(db_path)

    try:
        show_tables(conn)
        menu(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()