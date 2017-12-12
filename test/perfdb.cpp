/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "miopen/db_record.hpp"
#include "temp_file_path.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {

struct TestData
{
    int x;
    int y;

    TestData()
    {
        const int seed = 42;

        static std::mt19937 rng(seed);
        static std::uniform_int_distribution<std::mt19937::result_type> dist{};

        x = dist(rng);
        y = dist(rng);
    }

    TestData(int x_, int y_) : x(x_), y(y_) {}

    void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = ',';
        TestData t;
        std::istringstream ss(s);

        auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

        if(!success)
            return false;

        *this = t;
        return true;
    }

    bool operator==(const TestData& other) const { return x == other.x && y == other.y; }

    private:
    inline static bool DeserializeField(std::istream& from, int* ret, char separator)
    {
        std::string part;

        if(!std::getline(from, part, separator))
            return false;

        const auto start = part.c_str();
        char* end;
        auto value = std::strtol(start, &end, 10);

        if(start == end)
            return false;

        *ret = value;
        return true;
    }
};

std::ostream& operator<<(std::ostream& s, const TestData& td)
{
    s << "x: " << td.x << ", y: " << td.y;
    return s;
}

class DbTest
{
    public:
    DbTest() : _temp_file_path("/tmp/miopen.tests.perfdb.XXXXXX") {}
    virtual ~DbTest() {}

    protected:
    static const TestData& key()
    {
        static const TestData data(1, 2);
        return data;
    }

    static const TestData& value0()
    {
        static const TestData data(3, 4);
        return data;
    }

    static const TestData& value1()
    {
        static const TestData data(5, 6);
        return data;
    }

    static const TestData& value2()
    {
        static const TestData data(7, 8);
        return data;
    }

    static const char* id0() { return "0"; }
    static const char* id1() { return "1"; }
    static const char* id2() { return "2"; }
    static const char* missing_id() { return "2"; }
    const char* temp_file_path() const { return _temp_file_path; }

    private:
    TempFilePath _temp_file_path;
};

class DbFindTest : public DbTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        boost::optional<DbRecord> record0, record1;
        TestData read0, read1;
        TestData invalid_key(100, 200);

        {
            Db db(temp_file_path());

            record0 = db.FindRecord(key());
            record1 = db.FindRecord(invalid_key);
        }

        EXPECT(record0);
        EXPECT(record0->GetValues(id0(), read0));
        EXPECT(record0->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
        EXPECT(!record1);
    }
};

class DbStoreTest : public DbTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path());

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file_path()), read).good());

        boost::optional<DbRecord> record_read;
        TestData read0, read1;

        {
            Db db(temp_file_path());

            record_read = db.FindRecord(key());
        }

        EXPECT(record_read);
        EXPECT(record_read->GetValues(id0(), read0));
        EXPECT(record_read->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbUpdateTest : public DbTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path());

        // Store record0 (key=id0:value0)
        DbRecord record0(key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        boost::optional<DbRecord> record_read;
        {
            Db db(temp_file_path());

            record_read = db.FindRecord(key());
        }

        EXPECT(record_read);
        EXPECT(record_read->GetValues(id0(), read0));
        EXPECT(record_read->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbRemoveTest : public DbTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path());

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record));
        }

        {
            Db db(temp_file_path());

            EXPECT(db.FindRecord(key()));
            EXPECT(db.RemoveRecord(key()));
            EXPECT(!db.FindRecord(key()));
        }
    }
};

class DbReadTest : public DbTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        TestData read0, read1;

        {
            Db db(temp_file_path());

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));
        }

        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbWriteTest : public DbTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path());

        {
            Db db(temp_file_path());

            EXPECT(db.Store(key(), id0(), value0()));
            EXPECT(db.Store(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file_path()), read).good());

        TestData read0, read1;

        {
            Db db(temp_file_path());

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));
        }

        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbOperationsTest : public DbTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path()); // To suppress warning in logs.

        TestData to_be_rewritten(7, 8);

        {
            Db db(temp_file_path());

            EXPECT(db.Store(key(), id0(), to_be_rewritten));
            EXPECT(db.Store(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Store(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Store(key(), id1(), value1()));
        }

        {
            Db db(temp_file_path());

            // Rewriting existing value to store it to file.
            EXPECT(db.Store(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing, read_missing_cmp(read_missing);
            Db db(temp_file_path());

            // Loading by id not present in record should execute well but return false as nothing
            // was read.
            EXPECT(!db.Load(key(), missing_id(), read_missing));

            // In such case value should not be changed.
            EXPECT_EQUAL(read_missing, read_missing_cmp);

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, value0());
            EXPECT_EQUAL(read1, value1());

            EXPECT(db.Remove(key(), id0()));

            read0 = read_missing_cmp;

            EXPECT(!db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());

            EXPECT(!db.Remove(key(), id0()));
        }

        {
            TestData read0, read1, read_missing_cmp(read0);
            Db db(temp_file_path());

            EXPECT(!db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());
        }
    }
};

class DbParallelTest : public DbTest
{
public:
    inline void Run()
    {
        {
            Db db(temp_file_path());
            EXPECT(db.Store(key(), id0(), value0()));
        }

        {
            Db db0(temp_file_path());
            Db db1(temp_file_path());

            auto r0 = db0.FindRecord(key());
            auto r1 = db0.FindRecord(key());

            EXPECT(r0);
            EXPECT(r1);

            EXPECT(r0->SetValues(id1(), value1()));
            EXPECT(r1->SetValues(id2(), value2()));

            EXPECT(db0.UpdateRecord(*r0));
            EXPECT(db1.UpdateRecord(*r1));
        }

        {
            Db db(temp_file_path());
            TestData read1, read2;

            EXPECT(db.Load(key(), id1(), read1));
            EXPECT(db.Load(key(), id2(), read2));

            EXPECT_EQUAL(read1, value1());
            EXPECT_EQUAL(read2, value2());
        }
    }
};

} // namespace tests
} // namespace miopen

int main()
{
    miopen::tests::DbFindTest().Run();
    miopen::tests::DbStoreTest().Run();
    miopen::tests::DbUpdateTest().Run();
    miopen::tests::DbRemoveTest().Run();
    miopen::tests::DbReadTest().Run();
    miopen::tests::DbWriteTest().Run();
    miopen::tests::DbOperationsTest().Run();
    miopen::tests::DbParallelTest().Run();

    return 0;
}
