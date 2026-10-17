#include "spindle/version.h"

#include <gtest/gtest.h>

TEST(Version, MatchesCMakeProjectVersion) { EXPECT_EQ(spindle::version(), "0.1.0"); }
